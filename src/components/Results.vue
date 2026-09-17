<template>
  <div class="results">
    <h2 class="chapter-title">Experimental Results</h2>
    <!-- 仿真结果：修改 src 换图片，修改 figcaption 换说明。 -->
    <section id="simulation-results" class="result-section">
      <h3>Simulation Results</h3>
      <figure>
        <img src="../assets/images/results/simulation-result.png" alt="SIMPLER simulation results" loading="lazy" />
        <figcaption>
          Across nine SIMPLER tasks, both <strong>PCD-Mask</strong> and
          <strong>PCD-Fast</strong> consistently improve three distinct robot policies.
          PCD-Fast increases the average success rate of OpenVLA, GR00T N1.7,
          and π₀ by <strong>41.6%</strong>, <strong>10.6%</strong>, and
          <strong>9.4%</strong>, respectively.
        </figcaption>
      </figure>
    </section>

    <!-- 真机结果：放好图片后，去掉 img 的注释，并移除占位 div。 -->
    <section id="real-world-results" class="result-section">
      <h3>Real-World Results</h3>
      <figure>
        <img src="../assets/images/web/results/realworld.png" alt="Real-world manipulation results" loading="lazy" />
        <figcaption>
          On five real-world manipulation tasks, both PCD variants substantially
          improve π₀.₅, increasing its average success rate from
          <strong>49.0%</strong> to <strong>72.0%</strong> with PCD-Mask and
          <strong>76.5%</strong> with PCD-Fast. PCD-Fast further reduces execution
          time compared with PCD-Mask across all five tasks, demonstrating a better
          performance-efficiency trade-off.
        </figcaption>
      </figure>
    </section>

  </div>
</template>

<style scoped>
.results { max-width: 1100px; margin: 0 auto; padding: 55px 30px 60px; }
.chapter-title { margin: 0; text-align: center; font-size: 32px; font-weight: 600; line-height: 1.25; }
.result-section { padding: 45px 0 30px; scroll-margin-top: 28px; }
h3 { margin: 0 0 30px; text-align: center; font-size: 26px; line-height: 1.25; font-weight: 600; }
figure { margin: 0; }
figure + figure { margin-top: 36px; }
img { display: block; width: 100%; height: auto; }
figcaption { margin: 18px auto 0; max-width: 980px; font-size: 18px; line-height: 1.7; text-align: left; }
.figure-placeholder { width: 100%; min-height: 220px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccd3dc; border-radius: 6px; background: #fafbfd; color: #8793a3; font-size: 16px; }
</style>
