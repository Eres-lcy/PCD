import { nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'

const BUFFER_TARGET_SECONDS = 2.5
const BUFFER_POLL_INTERVAL_MS = 120
const PLAY_RETRY_DELAY_MS = 1500

export function useSynchronizedVideoGroups({ pages, pageIndex, tasks }) {
  const allTasks = pages.flatMap(page => page.tasks)
  const state = reactive(Object.fromEntries(allTasks.map(task => [task.key, 'loading'])))
  const groups = Object.fromEntries(allTasks.map(task => [task.key, {
    players: {},
    ready: new Set(),
    ended: new Set(),
    retryTimer: null,
    bufferTimer: null,
    operation: 0,
    primed: false,
  }]))
  const sectionElement = ref(null)
  const sectionActivated = ref(false)
  const sectionVisible = ref(false)
  let disposed = false
  let generation = 0
  let activationObserver = null
  let visibilityObserver = null

  function clearGroupTimers(group) {
    clearTimeout(group.retryTimer)
    clearTimeout(group.bufferTimer)
    group.retryTimer = null
    group.bufferTimer = null
  }

  function cancelGroupOperation(group) {
    group.operation++
    clearGroupTimers(group)
  }

  function changePage(direction) {
    generation++
    Object.values(groups).forEach(group => {
      cancelGroupOperation(group)
      Object.values(group.players).forEach(video => video.pause())
      group.players = {}
      group.ready.clear()
      group.ended.clear()
      group.primed = false
    })
    Object.keys(state).forEach(key => { state[key] = 'loading' })
    pageIndex.value = (pageIndex.value + direction + pages.length) % pages.length
  }

  function isCurrentTask(task) {
    return !disposed && tasks.value.some(item => item.key === task)
  }

  function isActive(task) {
    return sectionVisible.value && isCurrentTask(task)
  }

  function setPlayer(task, method, element) {
    if (element) groups[task].players[method] = element
  }

  function bufferedAhead(video) {
    const currentTime = video.currentTime
    for (let index = 0; index < video.buffered.length; index++) {
      if (video.buffered.start(index) <= currentTime + 0.05 && video.buffered.end(index) >= currentTime) {
        return Math.max(0, video.buffered.end(index) - currentTime)
      }
    }
    return 0
  }

  function hasEnoughBuffer(video) {
    if (video.ended) return true
    if (!Number.isFinite(video.duration)) return false
    const remaining = Math.max(0, video.duration - video.currentTime)
    if (remaining <= 0.15) return true
    const required = Math.min(BUFFER_TARGET_SECONDS, Math.max(0.1, remaining - 0.1))
    return bufferedAhead(video) + 0.05 >= required
  }

  function operationIsCurrent(task, operation, startedGeneration) {
    return !disposed && startedGeneration === generation && isActive(task) && groups[task].operation === operation
  }

  function waitForBuffer(task, players, operation, startedGeneration) {
    const group = groups[task]
    return new Promise(resolve => {
      const check = () => {
        if (!operationIsCurrent(task, operation, startedGeneration)) {
          resolve(false)
          return
        }
        if (players.some(video => video.error)) {
          resolve(false)
          return
        }
        if (players.every(hasEnoughBuffer)) {
          group.bufferTimer = null
          resolve(true)
          return
        }
        group.bufferTimer = setTimeout(check, BUFFER_POLL_INTERVAL_MS)
      }
      check()
    })
  }

  function seekTo(video, targetTime) {
    const duration = Number.isFinite(video.duration) ? video.duration : targetTime
    const target = Math.max(0, Math.min(targetTime, Math.max(0, duration - 0.05)))
    if (!video.seeking && Math.abs(video.currentTime - target) < 0.03) return Promise.resolve()

    return new Promise(resolve => {
      let timeoutId = null
      const finish = () => {
        clearTimeout(timeoutId)
        video.removeEventListener('seeked', finish)
        resolve()
      }
      video.addEventListener('seeked', finish, { once: true })
      timeoutId = setTimeout(finish, 2000)
      try {
        video.currentTime = target
      } catch {
        finish()
      }
    })
  }

  function scheduleRetry(task, callback) {
    const group = groups[task]
    clearTimeout(group.retryTimer)
    state[task] = 'waiting'
    group.retryTimer = setTimeout(() => {
      if (isActive(task) && state[task] === 'waiting') callback()
    }, PLAY_RETRY_DELAY_MS)
  }

  async function playPreparedGroup(task, players, operation, startedGeneration, retry) {
    if (!operationIsCurrent(task, operation, startedGeneration)) return false

    const results = await Promise.allSettled(players.map(video => video.play()))
    if (!operationIsCurrent(task, operation, startedGeneration)) {
      players.forEach(video => video.pause())
      return false
    }
    if (state[task] === 'error' || results.some(result => result.status === 'rejected')) {
      players.forEach(video => video.pause())
      scheduleRetry(task, retry)
      return false
    }

    state[task] = 'playing'
    return true
  }

  async function restart(task) {
    if (!isActive(task) || ['starting', 'buffering'].includes(state[task])) return
    const group = groups[task]
    const players = Object.values(group.players)
    if (players.length !== 3 || players.some(video => video.error)) return

    cancelGroupOperation(group)
    const operation = group.operation
    const startedGeneration = generation
    state[task] = 'starting'
    group.ended.clear()

    players.forEach(video => {
      video.pause()
      video.muted = true
      video.defaultMuted = true
      video.playbackRate = 1
    })
    await Promise.all(players.map(video => seekTo(video, 0)))
    if (!operationIsCurrent(task, operation, startedGeneration)) return

    const buffered = await waitForBuffer(task, players, operation, startedGeneration)
    if (!buffered) return
    await playPreparedGroup(task, players, operation, startedGeneration, () => restart(task))
  }

  async function resume(task) {
    if (!isActive(task) || state[task] === 'error') return
    const group = groups[task]
    const entries = Object.entries(group.players)
    if (entries.length !== 3 || entries.some(([, video]) => video.error)) return

    entries.forEach(([method, video]) => {
      if (video.readyState >= 3) group.ready.add(method)
    })
    if (group.ready.size !== 3) return
    if (['loading', 'waiting'].includes(state[task])) {
      restart(task)
      return
    }

    const activePlayers = entries
      .filter(([method, video]) => !group.ended.has(method) && !video.ended)
      .map(([, video]) => video)
    if (activePlayers.length === 0) {
      restart(task)
      return
    }

    cancelGroupOperation(group)
    const operation = group.operation
    const startedGeneration = generation
    state[task] = 'starting'
    const targetTime = Math.min(...activePlayers.map(video => video.currentTime))
    activePlayers.forEach(video => video.pause())
    await Promise.all(activePlayers.map(video => seekTo(video, targetTime)))
    if (!operationIsCurrent(task, operation, startedGeneration)) return

    const buffered = await waitForBuffer(task, activePlayers, operation, startedGeneration)
    if (!buffered) return
    await playPreparedGroup(task, activePlayers, operation, startedGeneration, () => resume(task))
  }

  async function onBuffering(task) {
    if (!isActive(task) || state[task] !== 'playing') return
    const group = groups[task]
    const activePlayers = Object.entries(group.players)
      .filter(([method, video]) => !group.ended.has(method) && !video.ended)
      .map(([, video]) => video)
    if (activePlayers.length === 0) return

    cancelGroupOperation(group)
    const operation = group.operation
    const startedGeneration = generation
    state[task] = 'buffering'
    activePlayers.forEach(video => video.pause())

    const targetTime = Math.min(...activePlayers.map(video => video.currentTime))
    await Promise.all(activePlayers.map(video => seekTo(video, targetTime)))
    if (!operationIsCurrent(task, operation, startedGeneration)) return

    const buffered = await waitForBuffer(task, activePlayers, operation, startedGeneration)
    if (!buffered) return
    await playPreparedGroup(task, activePlayers, operation, startedGeneration, () => resume(task))
  }

  function primeGroup(task) {
    const group = groups[task]
    const entries = Object.entries(group.players)
    if (entries.length !== 3) return

    entries.forEach(([method, video]) => {
      video.preload = 'auto'
      if (video.readyState >= 3) group.ready.add(method)
    })
    if (!group.primed) {
      group.primed = true
      entries.forEach(([, video]) => {
        if (video.readyState < 3) video.load()
      })
    }
    if (group.ready.size === 3) resume(task)
  }

  function onReady(task, method) {
    if (!isCurrentTask(task)) return
    groups[task].ready.add(method)
    if (sectionVisible.value && groups[task].ready.size === 3 && ['loading', 'waiting'].includes(state[task])) {
      restart(task)
    }
  }

  function pauseForVisibility() {
    for (const task of tasks.value) {
      const group = groups[task.key]
      cancelGroupOperation(group)
      Object.values(group.players).forEach(video => video.pause())
      if (['starting', 'playing', 'waiting', 'buffering'].includes(state[task.key])) {
        state[task.key] = 'paused'
      }
    }
  }

  function resumeVisibleGroups() {
    for (const task of tasks.value) primeGroup(task.key)
  }

  function onEnded(task, method) {
    if (!isCurrentTask(task)) return
    const group = groups[task]
    group.ended.add(method)
    if (isActive(task) && group.ended.size === 3 && state[task] === 'playing') restart(task)
  }

  function onError(task) {
    if (!isCurrentTask(task)) return
    const group = groups[task]
    cancelGroupOperation(group)
    state[task] = 'error'
    Object.values(group.players).forEach(video => video.pause())
  }

  onMounted(() => {
    activationObserver = new IntersectionObserver(entries => {
      if (entries[0]?.isIntersecting) {
        sectionActivated.value = true
        activationObserver?.disconnect()
      }
    }, { rootMargin: '400px 0px' })

    visibilityObserver = new IntersectionObserver(async entries => {
      const entry = entries[0]
      const visible = Boolean(entry?.isIntersecting && entry.intersectionRatio >= 0.05)
      if (visible === sectionVisible.value) return
      sectionVisible.value = visible
      if (visible) {
        sectionActivated.value = true
        await nextTick()
        resumeVisibleGroups()
      } else {
        pauseForVisibility()
      }
    }, { threshold: [0, 0.05] })

    if (sectionElement.value) {
      activationObserver.observe(sectionElement.value)
      visibilityObserver.observe(sectionElement.value)
    }
  })

  onBeforeUnmount(() => {
    disposed = true
    activationObserver?.disconnect()
    visibilityObserver?.disconnect()
    Object.values(groups).forEach(group => {
      cancelGroupOperation(group)
      Object.values(group.players).forEach(video => video.pause())
    })
  })

  return {
    changePage,
    onBuffering,
    onEnded,
    onError,
    onReady,
    sectionActivated,
    sectionElement,
    setPlayer,
    state,
  }
}
