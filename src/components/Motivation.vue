<script setup>
import motivationImage from '../assets/images/web/motivation.png'
</script>


<template>
  <section class="motivation-section">

    <div class="motivation-content">

      <div class="motivation-figure">
        <img
          :src="motivationImage"
          width="1200"
          height="1033"
          loading="lazy"
          decoding="async"
          alt="Motivation of Policy Contrastive Decoding"
        />
      </div>

      <div class="motivation-text">

        <h2 class="section-title">
          Motivation
        </h2>

        <p class="motivation-description">
          Existing robot policies are prone to learning
          <strong>spurious correlations</strong>
          from pre-training trajectories, causing them to rely on
          task-irrelevant visual features rather than object-relevant cues
          for action prediction. Consequently, even small changes in the
          visual environment can lead to substantial performance
          degradation. As illustrated on the left, changing the lighting
          conditions (<strong>(a)</strong> to <strong>(b)</strong>) or shifting the drawer handle position (<strong>(a)</strong> to <strong>(c)</strong>) significantly
          reduces the success rate of the baseline policy, revealing its
          limited generalization to unseen scenarios.
        </p>

      </div>

    </div>

  </section>
</template>


<style scoped>
.motivation-section {
  max-width: 1100px;
  margin: 0 auto;
  padding: 45px 30px 60px;
}


.motivation-content {
  display: flex;
  align-items: flex-end;

  gap: 45px;
}


.motivation-figure {
  flex: 0.75;
}


.motivation-figure img {
  display: block;

  width: 100%;
  height: auto;

  margin: 0 auto;
}


.motivation-text {
  flex: 1;

  text-align: left;
}


.section-title {
  text-align: center;

  margin: 0 0 20px;

  font-size: 32px;
  line-height: 1.2;
  font-weight: 600;
}


.motivation-description {
  margin: 0;

  font-size: 18px;
  line-height: 1.7;
}
</style>
