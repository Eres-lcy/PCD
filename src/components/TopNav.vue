<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'

const links = [
  { id: 'demos', label: 'Demos' },
  { id: 'overview', label: 'Overview' },
  { id: 'method', label: 'Framework' },
  { id: 'results', label: 'Results' },
  { id: 'robustness', label: 'Robustness' },
]
const active = ref('')
let frame = 0
function updateActive() {
  frame = 0
  active.value = links.filter(link => document.getElementById(link.id)?.getBoundingClientRect().top <= 120).at(-1)?.id || ''
}
function onScroll() {
  if (!frame) frame = requestAnimationFrame(updateActive)
}
onMounted(() => {
  updateActive()
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll)
})
onBeforeUnmount(() => {
  cancelAnimationFrame(frame)
  window.removeEventListener('scroll', onScroll)
  window.removeEventListener('resize', onScroll)
})
</script>

<template>
  <header class="site-header">
    <nav aria-label="Main navigation" class="navigation">
      <a class="brand" href="#top" aria-label="PCD — back to top">PCD<span>Policy Contrastive Decoding</span></a>
      <div class="nav-links">
        <a v-for="link in links" :key="link.id" :href="`#${link.id}`"
          :class="{ active: active === link.id }" :aria-current="active === link.id ? 'location' : undefined">{{ link.label }}</a>
      </div>
    </nav>
  </header>
</template>

<style scoped>
.site-header { position: sticky; top: 0; z-index: 20; background: rgba(255,255,255,.96); border-bottom: 1px solid #e8edf4; backdrop-filter: blur(12px); }
.navigation { max-width: 1350px; height: 64px; margin: 0 auto; padding: 0 30px; display: flex; align-items: center; justify-content: space-between; gap: 32px; }
a { text-decoration: none; }
.brand { display: flex; align-items: center; gap: 15px; color: #3273dc; font-size: 23px; font-weight: 750; }
.brand span { color: #788394; font-size: 13px; font-weight: 400; }
.nav-links { display: flex; align-items: center; gap: 8px; }
.nav-links a { padding: 9px 14px; border-radius: 7px; color: #566174; font-size: 14px; font-weight: 500; transition: background .15s, color .15s; }
.nav-links a:hover, .nav-links a.active { color: #2763bc; background: #edf3fd; }
</style>
