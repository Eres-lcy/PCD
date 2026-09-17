<script setup>
import Hero from './components/Hero.vue'
import Highlights from './components/Highlights.vue'
import Abstract from './components/Abstract.vue'
import Motivation from './components/Motivation.vue'
import Method from './components/Method.vue'
import Results from './components/Results.vue'
import Demo from './components/Demo.vue'
import OodDemo from './components/OodDemo.vue'
import TopNav from './components/TopNav.vue'
import Footer from './components/Footer.vue'
</script>


<template>
  <TopNav />
  <main id="top" class="page">
    <Hero />
    <Highlights />
    <div class="chapter-band"><Demo /></div>
    <div id="overview" class="chapter-overview">
      <Abstract />
      <Motivation />
    </div>
    <div id="method" class="chapter-band"><Method /></div>
    <div id="results"><Results /></div>
    <div id="robustness" class="chapter-band"><OodDemo /></div>
  </main>
  <Footer />
</template>
