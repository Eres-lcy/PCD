<script setup>
import frameworkImage from '../assets/images/web/framework.png'
import maskImage from '../assets/images/web/PCD-Mask.png'
import fastImage from '../assets/images/web/PCD-Fast.png'
</script>


<template>
  <section class="method-section">

    <h2 class="section-title">
      Framework
    </h2>


    <!-- Unified PCD framework -->
    <div class="framework-block">

      <h3 class="framework-title">
        Policy Contrastive Decoding
      </h3>

      <div class="framework-figure">
        <img
          :src="frameworkImage"
          width="2200"
          height="756"
          loading="lazy"
          decoding="async"
          alt="Overview of the Policy Contrastive Decoding framework"
        />
      </div>

      <div class="framework-description">
        <p>
          <strong>Policy Contrastive Decoding (PCD)</strong> is a unified,
          <strong>training-free</strong> framework for mitigating spurious correlations in
          robotic foundation models. Given an observation and a language
          instruction, PCD contrasts action predictions derived from the
          original representation with those from an
          <strong>object-ignored representation</strong>, thereby amplifying
          object-relevant decision factors while suppressing the influence
          of spurious visual cues. Specifically, we develop two complementary PCD variants:
          <strong>PCD-Mask</strong> and <strong>PCD-Fast</strong>,
          which differ in how the object-ignored representation is constructed.
        </p>
      </div>

    </div>


    <!-- PCD-Mask -->
    <section
      class="variant-block mask-block"
      aria-labelledby="pcd-mask-title"
    >

      <figure class="variant-figure">
        <img
          :src="maskImage"
          width="1200"
          height="708"
          loading="lazy"
          decoding="async"
          alt="PCD-Mask uses Track2Mask to construct an object-masked observation and contrasts the policy predictions from the original and masked observations."
        />
        <figcaption>
          PCD-Mask constructs object-ignored representations through visual masking.
        </figcaption>
      </figure>


      <div class="variant-copy">

        <h3
          id="pcd-mask-title"
          class="variant-title"
        >
          PCD-Mask (Conference Version)
        </h3>

        <div class="variant-description">

          <p>
            <strong>PCD-Mask</strong> constructs the object-ignored representation
            directly in the observation space.
            <strong>Track2Mask</strong> first localizes the instruction-specified
            target object, tracks and segments it throughout the trajectory, and
            inpaints the segmented regions to obtain object-masked observations.
          </p>

          <p>
            The same pretrained policy processes both the original and object-masked
            observations, and PCD contrasts the resulting action predictions to
            amplify object-relevant decision cues. Autoregressive policies use their
            native action probabilities, whereas <strong>KDE-PM</strong> approximates
            action distributions for flow-matching policies. Because both mechanisms
            rely only on policy outputs, PCD-Mask requires neither access to internal
            activations nor modification of model parameters, enabling
            <strong>black-box policy access</strong>.
          </p>

        </div>

      </div>

    </section>


    <!-- PCD-Fast -->
    <section
      class="variant-block fast-block"
      aria-labelledby="pcd-fast-title"
    >

      <div class="variant-copy">

        <h3
          id="pcd-fast-title"
          class="variant-title"
        >
          PCD-Fast
        </h3>

        <div class="variant-description">

          <p>
            <strong>PCD-Fast</strong> constructs the object-ignored representation
            directly from intermediate policy features. It reuses features from the
            original computation and performs contrastive decoding in
            <strong>a single policy forward pass</strong>, eliminating observation
            inpainting and an additional policy evaluation.
            <strong>Adaptive Contrastive Layer Selection (ACLS)</strong> identifies
            an intermediate representation with reduced target-object reliance while
            preserving meaningful action information.
          </p>

          <p>
            The selected intermediate representation and the final representation
            are decoded through the policy's original output projector and contrasted
            within the same forward pass. Autoregressive policies directly contrast
            their action distributions, while <strong>Contrastive Vector-field
            Decoding (CVD)</strong> performs the same contrastive principle directly 
            in the native vector-field space of flow-matching policies.
          </p>

        </div>

      </div>


      <figure class="variant-figure">
        <img
          :src="fastImage"
          width="1200"
          height="699"
          loading="lazy"
          decoding="async"
          alt="PCD-Fast adaptively selects an intermediate Transformer layer and contrasts its projected prediction with the final-layer prediction within the same policy forward pass."
        />
        <figcaption>
          PCD-Fast constructs object-ignored representations directly from intermediate policy features.
        </figcaption>
      </figure>

    </section>

  </section>
</template>


<style scoped>
.method-section {
  max-width: 1100px;
  margin: 0 auto;
  padding: 45px 30px 60px;
}


/* =========================
   Section title
========================= */

.section-title {
  margin: 0 0 34px;

  text-align: center;

  font-size: 32px;
  line-height: 1.2;
  font-weight: 600;
}


/* =========================
   Unified framework
========================= */

.framework-block {
  width: 100%;
}


.framework-title {
  margin: 0 0 24px;

  text-align: center;

  font-size: 24px;
  line-height: 1.3;
  font-weight: 600;
}


.framework-figure {
  width: 100%;
  margin: 0 auto;
}


.framework-figure img {
  display: block;

  width: 100%;
  height: auto;

  margin: 0 auto;
}


.framework-description {
  max-width: 1100px;

  margin: 28px auto 0;

  font-size: 18px;
  line-height: 1.7;

  text-align: left;
}


.framework-description p {
  margin: 0;
}


/* =========================
   PCD variants
========================= */

.variant-block {
  display: grid;

  align-items: center;

  gap: 44px;

  margin-top: 64px;
}


.mask-block {
  grid-template-columns:
    minmax(0, 0.85fr)
    minmax(0, 1.15fr);
}


.fast-block {
  grid-template-columns:
    minmax(0, 1.15fr)
    minmax(0, 0.85fr);
}


/* =========================
   Variant figures
========================= */

.variant-figure {
  margin: 0;
  min-width: 0;
}


.variant-figure img {
  display: block;

  width: 100%;
  height: auto;
}


.variant-figure figcaption {
  margin-top: 12px;

  color: #384b73;

  text-align: center;

  font-size: 16px;
  line-height: 1.5;
}


/* =========================
   Variant text
========================= */

.variant-copy {
  min-width: 0;
}


.variant-title {
  margin: 0 0 8px;

  font-size: 26px;
  line-height: 1.3;
  font-weight: 600;
}


.variant-description {
  font-size: 17px;
  line-height: 1.7;

  color: #3f4349;
}


.variant-description p {
  margin: 0 0 18px;
}


.variant-description p:last-child {
  margin-bottom: 0;
}


.variant-description strong {
  font-weight: 600;
}
</style>
