<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'

// Videos are stored inside the project and bundled by Vite.
const videos = import.meta.glob('../assets/videos/{realworld_demo_2x,simulation_demo}/*.mp4', {
  eager: true, query: '?url', import: 'default',
})

// Execution times in seconds from the paper, Fig. 5 (not the 2× video duration).
const realWorldTasks = [
  { id: 'brush', title: 'Store Brush', mask: '23.0', fast: '14.9' },
  { id: 'pen', title: 'Store Markers', mask: '21.3', fast: '18.7' },
  { id: 'can', title: 'Pick up Can', mask: '8.9', fast: '5.8' },
  { id: 'towel', title: 'Fold Towel', mask: '17.8', fast: '16.0' },
  { id: 'cube', title: 'Place Cube', mask: '12.4', fast: '10.3' },
]
const pages = [
  { title: 'Real-World Demonstrations', folder: 'realworld_demo_2x', speed: '2×', ratio: '498 / 360', tasks: realWorldTasks },
  { title: 'Simulation(SimplerEnv) Demonstrations', folder: 'simulation_demo', speed: '1×', ratio: '640 / 512', tasks: [
    { id: 'close_drawer', title: 'Close Drawer' },
    { id: 'move_near', title: 'Move Near' },
    { id: 'open_drawer', title: 'Open Drawer' },
    { id: 'pick_can', title: 'Coke Can' },
    { id: 'place_apple', title: 'Apple' },
  ] }, 
  { title: 'Simulation(SimplerEnv) Demonstrations', folder: 'simulation_demo', speed: '1×', ratio: '640 / 480', tasks: [
    { id: 'put_carrot', title: 'Carrot Plate' },
    { id: 'eggplant', title: 'Eggplant Basket' },
    { id: 'spoon', title: 'Spoon Towel' },
    { id: 'cube', title: 'Stack Cube' },
  ] },
]
pages.forEach((page, index) => page.tasks.forEach(task => { task.key = `${index}-${task.id}` }))
const pageIndex = ref(0)
const currentPage = computed(() => pages[pageIndex.value])
const tasks = computed(() => currentPage.value.tasks)
const methods = computed(() => [
  { id: 'baseline', label: pageIndex.value === 0 ? 'Baseline (π₀.₅)' : 'Baseline (π₀)', file: 'baseline', outcome: 'failure', outcomeLabel: 'Failed', outcomeSymbol: '✕' },
  { id: 'mask', label: '+ PCD-Mask', file: 'pcd', outcome: 'success', outcomeLabel: 'Success', outcomeSymbol: '✓' },
  { id: 'fast', label: '+ PCD-Fast', file: 'plcd', outcome: 'success', outcomeLabel: 'Success', outcomeSymbol: '✓' },
])
const allTasks = pages.flatMap(page => page.tasks)
const state = reactive(Object.fromEntries(allTasks.map(task => [task.key, 'loading'])))
const groups = Object.fromEntries(allTasks.map(task => [task.key, {
  players: {}, ready: new Set(), ended: new Set(), retryTimer: null,
}]))
const sectionElement = ref(null)
const sectionActivated = ref(false)
const sectionVisible = ref(false)
let disposed = false
let generation = 0
let activationObserver = null
let visibilityObserver = null

function changePage(direction) {
  generation++
  Object.values(groups).forEach(group => {
    clearTimeout(group.retryTimer)
    Object.values(group.players).forEach(video => video.pause())
    group.players = {}
    group.ready.clear()
    group.ended.clear()
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

function videoSource(task, method) {
  return videos[`../assets/videos/${currentPage.value.folder}/${task.id}_${method.file}.mp4`]
}

function setPlayer(task, method, element) {
  if (element) groups[task].players[method] = element
}

function onReady(task, method) {
  if (!isCurrentTask(task)) return
  groups[task].ready.add(method)
  if (sectionVisible.value && groups[task].ready.size === 3 && state[task] === 'loading') restart(task)
}

async function restart(task) {
  if (!isActive(task) || state[task] === 'starting') return
  const startedGeneration = generation
  const group = groups[task]
  const players = Object.values(group.players)
  if (players.length !== 3 || players.some(video => video.error)) return
  clearTimeout(group.retryTimer)
  state[task] = 'starting'
  group.ended.clear()
  for (const video of players) {
    video.pause()
    video.currentTime = 0
    video.muted = true
    video.defaultMuted = true
    video.playbackRate = 1 // Real-world files are already 2×; simulation files stay at 1×.
  }
  // Start all three in the same turn; play() waits for each video's seek to finish.
  const results = await Promise.allSettled(players.map(video => video.play()))
  if (disposed || startedGeneration !== generation) return
  if (!sectionVisible.value) {
    players.forEach(video => video.pause())
    if (state[task] !== 'error') state[task] = 'paused'
    return
  }
  if (state[task] === 'error' || results.some(result => result.status === 'rejected')) {
    players.forEach(video => video.pause())
    if (state[task] !== 'error') {
      state[task] = 'waiting'
      // Retry a temporarily interrupted start without requiring a manual button.
      group.retryTimer = setTimeout(() => restart(task), 1500)
    }
  } else {
    state[task] = 'playing'
  }
}

async function resume(task) {
  if (!isActive(task) || state[task] === 'error') return
  const startedGeneration = generation
  const group = groups[task]
  const entries = Object.entries(group.players)
  if (entries.length !== 3 || entries.some(([, video]) => video.error)) return

  entries.forEach(([method, video]) => {
    if (video.readyState >= 3) group.ready.add(method)
  })
  if (group.ready.size !== 3) return
  if (state[task] === 'loading' || state[task] === 'waiting') {
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

  state[task] = 'starting'
  const results = await Promise.allSettled(activePlayers.map(video => video.play()))
  if (disposed || startedGeneration !== generation) return
  if (!sectionVisible.value) {
    activePlayers.forEach(video => video.pause())
    if (state[task] !== 'error') state[task] = 'paused'
    return
  }
  if (results.some(result => result.status === 'rejected')) {
    activePlayers.forEach(video => video.pause())
    state[task] = 'waiting'
    clearTimeout(group.retryTimer)
    group.retryTimer = setTimeout(() => resume(task), 1500)
  } else {
    state[task] = 'playing'
  }
}

function pauseForVisibility() {
  for (const task of tasks.value) {
    const group = groups[task.key]
    clearTimeout(group.retryTimer)
    Object.values(group.players).forEach(video => video.pause())
    if (['starting', 'playing', 'waiting'].includes(state[task.key])) {
      state[task.key] = 'paused'
    }
  }
}

function resumeVisibleGroups() {
  for (const task of tasks.value) {
    const group = groups[task.key]
    Object.entries(group.players).forEach(([method, video]) => {
      if (video.readyState >= 3) group.ready.add(method)
    })
    if (group.ready.size === 3) resume(task.key)
  }
}

function onEnded(task, method) {
  if (!isCurrentTask(task)) return
  // A finished video holds its final frame while its two peers continue.
  groups[task].ended.add(method)
  if (isActive(task) && groups[task].ended.size === 3 && state[task] === 'playing') restart(task)
}

function onError(task) {
  if (!isCurrentTask(task)) return
  clearTimeout(groups[task].retryTimer)
  state[task] = 'error'
  Object.values(groups[task].players).forEach(video => video.pause())
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
    clearTimeout(group.retryTimer)
    Object.values(group.players).forEach(video => video.pause())
  })
})
</script>

<template>
  <section id="demos" ref="sectionElement" class="demo-section" aria-labelledby="demo-title">
    <div class="demo-heading">
      <h2 id="demo-title" aria-live="polite">{{ currentPage.title }}</h2>
    </div>
    <p class="page-number">{{ pageIndex + 1 }} / {{ pages.length }}</p>
    <div class="demo-carousel">
      <button type="button" class="page-arrow carousel-arrow arrow-previous" aria-label="Previous demo page" @click="changePage(-1)">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 4 L8 12 L16 20" /></svg>
      </button>
      <div class="demo-table-wrapper">
        <table :key="pageIndex" class="demo-table" :class="{ 'four-tasks': tasks.length === 4 }"
          :style="{ '--video-ratio': currentPage.ratio }"
          :aria-label="`${currentPage.title}: ${tasks.length} tasks compared across Baseline, PCD-Mask, and PCD-Fast`">
          <thead>
            <tr>
              <td class="method-column"></td>
              <th v-for="task in tasks" :key="task.id" scope="col">
                {{ task.title }}
                <span v-if="state[task.key] === 'error'" class="video-error" role="status">Video unavailable</span>
              </th>
            </tr>
          </thead>
          <tbody v-for="method in methods" :key="method.id" :class="`row-${method.id}`">
            <tr class="video-row">
              <th scope="row" class="method-label">{{ method.label }}</th>
              <td v-for="task in tasks" :key="task.id">
                <div v-if="sectionActivated" class="video-shell">
                  <video :ref="element => setPlayer(task.key, method.id, element)"
                    :src="videoSource(task, method)"
                    :aria-label="`${task.title} — ${method.label}, ${method.outcomeLabel}, ${currentPage.speed} speed`"
                    :data-task="task.id" :data-method="method.id"
                    muted playsinline preload="metadata" width="498" height="360"
                    @canplay="onReady(task.key, method.id)" @ended="onEnded(task.key, method.id)"
                    @error="onError(task.key)" />
                  <span class="outcome-badge" :class="`outcome-${method.outcome}`">
                    <span aria-hidden="true">{{ method.outcomeSymbol }}</span>
                    {{ method.outcomeLabel }}
                  </span>
                </div>
              </td>
            </tr>
            <tr class="time-row" :aria-hidden="method.id === 'baseline' || pageIndex !== 0 ? 'true' : undefined">
              <th scope="row" class="execution-label">
                <span v-if="method.id !== 'baseline' && pageIndex === 0">Execution Time</span>
                <span v-else>&nbsp;</span>
              </th>
              <td v-for="task in tasks" :key="task.id" class="execution-time">
                <span v-if="method.id !== 'baseline' && pageIndex === 0" class="time-value">
                  <strong>{{ task[method.id] }}</strong><span class="time-unit">s</span>
                </span>
                <span v-else>&nbsp;</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <button type="button" class="page-arrow carousel-arrow arrow-next" aria-label="Next demo page" @click="changePage(1)">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4 L16 12 L8 20" /></svg>
      </button>
    </div>
    <p v-if="pageIndex === 0" class="demo-note">
      Both PCD-Mask and PCD-Fast improve task success rate over the baseline, while PCD-Fast achieves lower execution time than PCD-Mask.
      (Videos are shown at {{ currentPage.speed }} speed.)
    </p>
    <p v-else class="demo-note">
      Both PCD-Mask and PCD-Fast improve task success rate over the baseline, while PCD-Fast achieves lower execution time than PCD-Mask.
    </p>
  </section>
</template>

<style scoped>
.demo-section { max-width: 1350px; margin: 0 auto; padding: 40px 30px 55px; }
h2 { margin: 0; text-align: center; font-size: 32px; line-height: 1.25; font-weight: 600; }
.demo-heading { text-align: center; }
.page-arrow { display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; width: 42px; height: 42px; padding: 0; border: 1px solid #dce2e9; border-radius: 50%; background: #fff; color: #4f6fae; cursor: pointer; }
.page-arrow svg { display: block; width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.page-arrow:hover { background: #f1f5fa; }
.page-arrow:focus-visible { outline: 2px solid #4f6fae; outline-offset: 3px; }
.page-number { margin: 10px 0 24px; text-align: center; color: #707782; font-size: 14px; }
.demo-carousel { position: relative; }
.carousel-arrow { position: absolute; top: 50%; z-index: 3; transform: translateY(-50%); box-shadow: 0 3px 12px rgba(49, 78, 122, .12); }
.arrow-previous { left: -50px; }
.arrow-next { right: -50px; }
.demo-description { margin: 0 0 28px; text-align: center; font-size: 18px; line-height: 1.6; color: #505762; }
.demo-table-wrapper { overflow-x: auto; container-type: inline-size; }
/* Use the five-column real-world video height as the shared row-center reference.
   229px = 145px label column + seven 12px horizontal table spaces. */
.demo-table { --row-video-height: calc((max(100cqw, 1050px) - 229px) / 5 * 360 / 498); width: 100%; min-width: 1050px; table-layout: fixed; border-collapse: separate; border-spacing: 12px 0; }
.demo-table.four-tasks { border-spacing: 40px 0; }
.four-tasks .method-label, .four-tasks .execution-label { transform: translateX(-28px); }
.method-column { width: 145px; }
thead th { padding: 0 0 16px; text-align: center; font-size: 17px; font-weight: 600; }
.method-label { padding: 0; text-align: left; vertical-align: middle; font-size: 18px; font-weight: 600; white-space: nowrap; }
td { padding: 0; vertical-align: top; }
.video-row th, .video-row td { height: var(--row-video-height); }
.video-row td { position: relative; }
.video-shell { position: absolute; top: 50%; width: 100%; aspect-ratio: var(--video-ratio); transform: translateY(-50%); }
video { display: block; width: 100%; height: 100%; object-fit: contain; background: #f4f6f8; border-radius: 5px; }
/* Badge position: increase top to move it down; increase right to move it left. */
.outcome-badge { position: absolute; top: 4px; right: 4px; z-index: 1; display: inline-flex; align-items: center; gap: 4px; padding: 5px 8px; border-radius: 999px; color: #fff; font-size: 12px; line-height: 1; font-weight: 650; letter-spacing: .01em; box-shadow: 0 1px 4px rgba(0,0,0,.2); }
.outcome-success { background: rgba(24, 121, 78, .94); }
.outcome-failure { background: rgba(180, 35, 24, .94); }
.time-row th, .time-row td { padding: 8px 0 24px; font-size: 15px; line-height: 26px; color: #596579; }
.execution-label { text-align: left; white-space: nowrap; font-weight: 600; letter-spacing: .01em; }
.execution-time { text-align: center; font-variant-numeric: tabular-nums; }
.time-value { display: inline-flex; align-items: baseline; justify-content: center; min-width: 62px; padding: 2px 10px; border: 1px solid #dce4ee; border-radius: 999px; background: #f5f7fa; color: #40536e; }
.time-value strong { font-size: 16px; font-weight: 700; }
.time-unit { margin-left: 3px; font-size: 12px; font-weight: 600; color: #78869a; }
.row-fast .time-value { border-color: #c8d9f3; background: #edf4ff; color: #315d9c; }
.row-fast .time-unit { color: #5d78a2; }
.demo-note { margin: 0 auto; max-width: 1000px; text-align: center; font-size: 20px; line-height: 1.65; }
.video-error { display: block; margin-top: 8px; color: #a04436; font-size: 13px; font-weight: 400; }
</style>
