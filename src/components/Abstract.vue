<template>
  <section class="abstract-section">
    <h2 class="section-title">Abstract</h2>

    <p class="abstract-text">
      Robotic foundation models, a.k.a. generalist robot policies, hold immense
      potential to enable flexible, general-purpose and dexterous robotic systems.
      Despite this, existing robot policies are prone to learning spurious
      correlations from pre-training trajectories, adversely affecting their
      generalization capabilities beyond the training data. To tackle this, we
      propose <strong>Policy Contrastive Decoding (PCD)</strong>, a novel framework that redirects
      the robot policy's focus toward object-relevant visual cues by contrasting
      action probability distributions derived from original and object-ignored
      representations. Depending on how the object-ignored representations are
      constructed, we instantiate PCD in two variants. PCD-Mask explicitly masks
      task-relevant image regions specified by the language instruction and performs
      contrastive decoding between the action distributions induced by the original
      and object-masked observations. In contrast, PCD-Fast adaptively extracts
      object-ignored representations from intermediate policy features, eliminating
      the additional forward pass required by PCD-Mask and achieving a better
      performance-efficiency trade-off. Notably, both variants are training-free
      and compatible with different types of robot policies. Extensive experiments
      in both simulation and real-world environments on four state-of-the-art
      policies—including the autoregressive OpenVLA and the flow-matching-based
      GR00T N1.7, π₀ and π₀.₅—validate the flexibility and effectiveness of our
      PCD framework.
    </p>
  </section>
</template>


<style scoped>
.abstract-section {
  max-width: 1100px;
  margin: 0 auto;
  padding: 45px 30px 55px;
}


.section-title {
  margin: 0 0 24px;

  text-align: center;

  font-size: 32px;
  line-height: 1.2;
  font-weight: 600;
}


.abstract-text {
  margin: 0;

  font-size: 18px;
  line-height: 1.75;
  font-weight: 400;

  color: #3f4349;

  text-align: justify;
}
</style>