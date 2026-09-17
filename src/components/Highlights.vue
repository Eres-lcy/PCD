<template>
  <section class="highlights">
    <div class="highlight-grid">

      <!-- 1. Unified framework -->
      <div class="highlight-card framework-card">
        <div class="framework-value">
          Unified Framework
        </div>

        <div class="framework-variants" aria-label="PCD framework variants">
          <div class="framework-variant">
            <strong>PCD-Mask</strong>
            <span>Visual Masking</span>
          </div>
          <div class="framework-variant">
            <strong>PCD-Fast</strong>
            <span>Single-Forward</span>
          </div>
        </div>
      </div>


      <!-- 2. Real-world results -->
      <div class="highlight-card">
        <div class="dual-values">
          <div class="method-result">
            <div class="result-value">+46.9%</div>
            <div class="method-name">PCD-Mask</div>
          </div>

          <div class="method-result">
            <div class="result-value">+56.1%</div>
            <div class="method-name">PCD-Fast</div>
          </div>
        </div>

        <div class="highlight-label">
          Real-World Success Gain (on π₀.₅)
        </div>

        <div class="highlight-sub">
        </div>
      </div>


      <!-- 3. Simulation results: rolling -->
      <div class="highlight-card simulation-card">

        <Transition name="fade" mode="out-in">
          <div
            :key="currentSimulation.policy"
            class="simulation-content"
          >
            <div class="policy-name">
              {{ currentSimulation.policy }}
            </div>

            <div class="dual-values">
              <div class="method-result">
                <div class="result-value">
                  {{ currentSimulation.mask }}
                </div>
                <div class="method-name">
                  PCD-Mask
                </div>
              </div>

              <div class="method-result">
                <div class="result-value">
                  {{ currentSimulation.fast }}
                </div>
                <div class="method-name">
                  PCD-Fast
                </div>
              </div>
            </div>
          </div>
        </Transition>

        <div class="highlight-label">
          Simulation Success Gain
        </div>

        <div class="simulation-dots">
          <span
            v-for="(_, index) in simulationResults"
            :key="index"
            class="dot"
            :class="{ active: index === simulationIndex }"
          ></span>
        </div>
      </div>

    </div>
  </section>
</template>


<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'


const simulationResults = [
  {
    policy: 'OpenVLA',
    mask: '+36.2%',
    fast: '+41.6%',
  },
  {
    policy: 'GR00T N1.7',
    mask: '+9.6%',
    fast: '+10.6%',
  },
  {
    policy: 'π₀',
    mask: '+8.8%',
    fast: '+9.4%',
  },
]


const simulationIndex = ref(0)

const currentSimulation = computed(() => {
  return simulationResults[simulationIndex.value]
})


let timer = null

onMounted(() => {
  timer = setInterval(() => {
    simulationIndex.value =
      (simulationIndex.value + 1) % simulationResults.length
  }, 2600)
})

onBeforeUnmount(() => {
  if (timer) {
    clearInterval(timer)
  }
})
</script>


<style scoped>
.highlights {
  max-width: 1100px;
  margin: 0 auto;
  padding: 10px 30px 50px;
}


.highlight-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}


/* -------------------------
   Card
------------------------- */

.highlight-card {
  min-height: 185px;
  padding: 22px 18px;

  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  text-align: center;

  background: #ffffff;

  border: 1px solid #e4e7ec;
  border-radius: 16px;

  box-shadow:
    0 2px 8px rgba(0, 0, 0, 0.05);

  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
}


.highlight-card:hover {
  transform: translateY(-4px);

  box-shadow:
    0 7px 20px rgba(0, 0, 0, 0.10);
}


/* -------------------------
   Main values
------------------------- */

.framework-value {
  max-width: 250px;

  font-size: 34px;
  line-height: 1.12;
  font-weight: 700;

  margin-bottom: 14px;
}


.framework-variants {
  position: relative;

  width: 100%;
  max-width: 255px;

  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));

  gap: 12px;
}


.framework-variants::after {
  content: '';

  position: absolute;
  top: 1px;
  bottom: 1px;
  left: 50%;

  width: 1px;

  background: rgba(255, 255, 255, 0.32);

  transform: translateX(-0.5px);
}


.framework-variant {
  display: flex;
  flex-direction: column;
  align-items: center;

  gap: 4px;

  color: rgba(255, 255, 255, 0.86);

  text-align: center;

  font-size: 15px;
  line-height: 1.35;
  font-weight: 400;
}


.framework-variant strong {
  color: #ffffff;

  font-size: 17px;
  font-weight: 650;
}


.framework-card .framework-value {
  color: #3273dc;
}


.framework-card .framework-variants::after {
  background: #d8dee8;
}


.framework-card .framework-variant {
  color: #707782;
}


.framework-card .framework-variant strong {
  color: #34435a;
}


.dual-values {
  width: 100%;

  display: flex;
  justify-content: center;
  align-items: flex-start;

  gap: 24px;

  margin-bottom: 13px;
}


.method-result {
  min-width: 82px;
}


.result-value {
  font-size: 28px;
  line-height: 1.15;
  font-weight: 700;

  color: #3273dc;
}


.method-name {
  margin-top: 5px;

  font-size: 18px;
  line-height: 1.2;
  font-weight: 500;

  color: #707782;
}


/* -------------------------
   Labels
------------------------- */

.highlight-label {
  font-size: 22px;
  line-height: 1.35;
  font-weight: 600;
}


.highlight-sub {
  margin-top: 7px;

  font-size: 20px;
  line-height: 1.35;

  color: #7b818b;
}


/* -------------------------
   Simulation rolling card
------------------------- */

.simulation-card {
  overflow: hidden;
}


.simulation-content {
  width: 100%;
}


.policy-name {
  margin-bottom: 10px;

  font-size: 24px;
  line-height: 1.2;
  font-weight: 600;

  color: #333333;
}


.simulation-dots {
  display: flex;
  justify-content: center;

  gap: 5px;

  margin-top: 9px;
}


.dot {
  width: 5px;
  height: 5px;

  border-radius: 50%;

  background: #c4c8cf;

  transition: all 0.25s ease;
}


.dot.active {
  width: 14px;
  border-radius: 6px;

  background: #3273dc;
}


/* -------------------------
   Transition
------------------------- */

.fade-enter-active,
.fade-leave-active {
  transition:
    opacity 0.22s ease,
    transform 0.22s ease;
}


.fade-enter-from {
  opacity: 0;
  transform: translateY(5px);
}


.fade-leave-to {
  opacity: 0;
  transform: translateY(-5px);
}
</style>
