<script setup>
import { computed, ref } from 'vue'
import { useSynchronizedVideoGroups } from '../composables/useSynchronizedVideoGroups'

// Videos are stored inside the project and bundled by Vite.
const videos = import.meta.glob('../assets/videos_web/{ood_real_videos_2x,ood_sim_videos}/*.mp4', {
  eager: true, query: '?url', import: 'default',
})

// Scene labels and file prefixes can be edited here.
const pages = [
  { title: 'Real-World Robustness Demonstrations', folder: 'ood_real_videos_2x', speed: '2×', ratio: '498 / 360', tasks: [
    { id: 'distractor', title: 'Distractors' },
    { id: 'spatial', title: 'Spatial Relation' },
    { id: 'brightness', title: 'Low Brightness' },
    { id: 'texture', title: 'Table Texture' },
    { id: 'background', title: 'Background' },
  ] },
  { title: 'Simulation (SimplerEnv) Robustness Demonstrations', folder: 'ood_sim_videos', speed: '1×', ratio: '640 / 512', tasks: [
    { id: 'light', title: 'Spatial Relation (Light)' },
    { id: 'drawer', title: 'Spatial Relation (Handle)' },
    { id: 'brightness', title: 'Low Brightness' },
    { id: 'texture1', title: 'Table Texture 1' },
    { id: 'texture2', title: 'Table Texture 2' },
  ] },
]
pages.forEach((page, index) => page.tasks.forEach(task => { task.key = `${index}-${task.id}` }))
const pageIndex = ref(0)
const currentPage = computed(() => pages[pageIndex.value])
const tasks = computed(() => currentPage.value.tasks)
const methods = computed(() => [
  { id: 'baseline', label: pageIndex.value === 0 ? 'Baseline (π₀.₅)' : 'Baseline', file: 'baseline', outcome: 'failure', outcomeLabel: 'Failed', outcomeSymbol: '✕' },
  { id: 'mask', label: '+ PCD-Mask', file: 'pcd', outcome: 'success', outcomeLabel: 'Success', outcomeSymbol: '✓' },
  { id: 'fast', label: '+ PCD-Fast', file: 'plcd', outcome: 'success', outcomeLabel: 'Success', outcomeSymbol: '✓' },
])
const {
  changePage,
  onBuffering,
  onEnded,
  onError,
  onReady,
  sectionActivated,
  sectionElement,
  setPlayer,
  state,
} = useSynchronizedVideoGroups({ pages, pageIndex, tasks })

function videoSource(task, method) {
  return videos[`../assets/videos_web/${currentPage.value.folder}/${task.id}_${method.file}.mp4`]
}
</script>

<template>
  <div class="robustness-chapter">
    <!-- OOD 泛化结果 -->
    <section id="robustness-results" class="result-section">
      <h2>Robustness to Diverse Spurious Correlations</h2>
      <figure>
        <img src="../assets/images/web/results/OOD-result.png" alt="Generalization under unseen scene variations" loading="lazy" />
        <figcaption>
          PCD consistently improves robustness under diverse unseen scene changes,
          including distractors, spatial relations, illumination, table textures,
          and backgrounds. The gains are particularly pronounced in the real world,
          where PCD-Fast retains <strong>64.5%-93.5%</strong> of its original
          performance, compared with only <strong>25%-65%</strong> for the
          π₀.₅ baseline.
        </figcaption>
      </figure>
    </section>

  <section id="ood-demos" ref="sectionElement" class="demo-section" aria-labelledby="ood-demo-title">
    <div class="demo-heading">
      <h3 id="ood-demo-title" aria-live="polite">{{ currentPage.title }}</h3>
    </div>
    <p class="page-number">{{ pageIndex + 1 }} / {{ pages.length }}</p>
    <div class="demo-carousel">
      <button type="button" class="page-arrow carousel-arrow arrow-previous" aria-label="Previous OOD demo page" @click="changePage(-1)">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 4 L8 12 L16 20" /></svg>
      </button>
      <div class="demo-table-wrapper">
        <table :key="pageIndex" class="demo-table"
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
                    :aria-label="`${task.title} — ${method.label}, ${method.outcomeLabel}`"
                    :data-task="task.id" :data-method="method.id"
                    muted playsinline preload="metadata"
                    @canplay="onReady(task.key, method.id)" @waiting="onBuffering(task.key)"
                    @stalled="onBuffering(task.key)" @ended="onEnded(task.key, method.id)"
                    @error="onError(task.key)" />
                  <span class="outcome-badge" :class="`outcome-${method.outcome}`">
                    <span aria-hidden="true">{{ method.outcomeSymbol }}</span>
                    {{ method.outcomeLabel }}
                  </span>
                </div>
              </td>
            </tr>
            <tr class="spacer-row" aria-hidden="true"><td colspan="6"></td></tr>
          </tbody>
        </table>
      </div>
      <button type="button" class="page-arrow carousel-arrow arrow-next" aria-label="Next OOD demo page" @click="changePage(1)">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4 L16 12 L8 20" /></svg>
      </button>
    </div>
    <p v-if="pageIndex === 0" class="demo-note">
      Across diverse unseen visual shifts, both PCD-Mask and PCD-Fast improve policy robustness, 
      enabling successful task completion in scenarios where the baseline policy fails. (Videos are shown at {{ currentPage.speed }} speed.)
    </p>
    <p v-else class="demo-note">
      Across diverse unseen visual shifts, both PCD-Mask and PCD-Fast improve policy robustness, 
      enabling successful task completion in scenarios where the baseline policy fails.
    </p>
  </section>
  </div>
</template>

<style scoped>
.robustness-chapter { padding-top: 55px; }
.result-section { max-width: 1100px; margin: 0 auto; padding: 0 30px 34px; }
.result-section h2 { margin: 0 0 30px; text-align: center; font-size: 32px; line-height: 1.25; font-weight: 600; }
.result-section figure { margin: 0; }
.result-section img { display: block; width: 100%; height: auto; }
.result-section figcaption { margin: 18px auto 0; max-width: 980px; font-size: 18px; line-height: 1.7; }
.demo-heading { position: relative; padding-top: 30px; }
.demo-heading::before { content: ''; position: absolute; top: 0; width: 72px; height: 2px; background: #d4ddeb; }

.demo-section { max-width: 1350px; margin: 0 auto; padding: 0 30px 35px; }
h3 { margin: 0; text-align: center; font-size: 26px; line-height: 1.25; font-weight: 600; }
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
.demo-table-wrapper { overflow-x: auto; container-type: inline-size; }
/* Use the real-world video height as the shared row-center reference.
   229px = 145px label column + seven 12px horizontal table spaces. */
.demo-table { --row-video-height: calc((max(100cqw, 1050px) - 229px) / 5 * 360 / 498); width: 100%; min-width: 1050px; table-layout: fixed; border-collapse: separate; border-spacing: 12px 0; }
.method-column { width: 145px; }
thead th { padding: 0 0 16px; text-align: center; font-size: 17px; font-weight: 600; }
.method-label { padding: 0; text-align: left; vertical-align: middle; font-size: 18px; font-weight: 600; white-space: nowrap; }
td { padding: 0; vertical-align: top; }
.video-row th, .video-row td { height: var(--row-video-height); }
.video-row td { position: relative; }
.video-shell { position: absolute; top: 50%; width: 100%; aspect-ratio: var(--video-ratio); transform: translateY(-50%); }
video { display: block; width: 100%; height: 100%; object-fit: contain; background: #f4f6f8; border-radius: 5px; }
.outcome-badge { position: absolute; top: 8px; right: 8px; z-index: 1; display: inline-flex; align-items: center; gap: 4px; padding: 5px 8px; border-radius: 999px; color: #fff; font-size: 12px; line-height: 1; font-weight: 650; letter-spacing: .01em; box-shadow: 0 1px 4px rgba(0,0,0,.2); }
.outcome-success { background: rgba(24, 121, 78, .94); }
.outcome-failure { background: rgba(180, 35, 24, .94); }
.spacer-row td { height: 40px; }
.demo-note { margin: 0 auto; max-width: 1000px; text-align: center; font-size: 20px; line-height: 1.65; }
.video-error { display: block; margin-top: 8px; color: #a04436; font-size: 13px; font-weight: 400; }
</style>
