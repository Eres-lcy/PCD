<template>
  <section class="hero">

    <h1 class="paper-title">
      Mitigating Spurious Correlations in Robotic Foundation Models
      via Training-free Policy Contrastive Decoding
    </h1>

    <div class="authors">
      <span class="author">
        Chengyun Li<sup>1</sup>,
      </span>

      <span class="author">
        Ji Zhang<sup>1</sup>,
      </span>

      <span class="author">
        Xu Luo<sup>1</sup>,
      </span>

      <span class="author">
        Lianli Gao<sup>1</sup>,
      </span>

      <span class="author">
        Shihan Wu<sup>2</sup>,
      </span>

      <span class="author">
        Pengpeng Zeng<sup>3</sup>,
      </span>

      <span class="author">
        and Jingkuan Song<sup>3</sup>
      </span>
    </div>

    <div class="institutions">
      <div>
        1. University of Electronic Science and Technology of China
      </div>

      <div>
        2. Beijing Academy of Artificial Intelligence
      </div>

      <div>
        3. Tongji University
      </div>
    </div>

    <div class="extension">
      Extended journal version of
      <a
        href="https://koorye.github.io/PCD/"
        target="_blank"
        rel="noopener noreferrer"
        class="conference-link"
      >
        “Policy Contrastive Decoding for Robotic Foundation Models”
      </a>
      (ICLR 2026)
    </div>

    <div class="resource-links" aria-label="Project resources">
      <a
        href="https://github.com/Eres-lcy/PCD"
        target="_blank"
        rel="noopener noreferrer"
        class="resource-button"
      >
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.53-.49-.33-.83-.82-1.05-.82-1.05-.67-.45.05-.44.05-.44.74.05 1.13.76 1.13.76.66 1.13 1.73.8 2.15.61.07-.48.26-.8.47-.98-1.6-.18-3.28-.8-3.28-3.56 0-.79.28-1.43.74-1.93-.07-.18-.32-.91.07-1.9 0 0 .6-.19 1.97.74A6.8 6.8 0 0 1 8 4.69c.61 0 1.23.08 1.8.24 1.36-.93 1.96-.74 1.96-.74.39.99.14 1.72.07 1.9.46.5.74 1.14.74 1.93 0 2.77-1.69 3.37-3.3 3.55.27.23.5.68.5 1.38 0 1-.01 1.81-.01 2.06 0 .21.14.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8" />
        </svg>
        Code
      </a>
    </div>

  </section>
</template>


<style scoped>
.hero {
  max-width: 1350px;
  margin: 0 auto;

  padding:
    72px
    30px
    38px;

  text-align: center;
}


.paper-title {
  max-width: 1250px;
  margin: 0 auto 36px;

  font-size: 52px;
  line-height: 1.15;
  font-weight: 600;
  letter-spacing: -0.5px;
}


.authors {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;

  gap: 10px;

  font-size: 22px;
  line-height: 1.7;

  margin-bottom: 14px;
}


.author {
  white-space: nowrap;
}


.authors sup {
  font-size: 0.65em;
  vertical-align: super;
  margin-left: 2px;
}


.institutions {
  font-size: 22px;
  line-height: 1.5;

  margin-bottom: 18px;
}

.author-note {
  margin-top: -6px;
  margin-bottom: 12px;

  font-size: 16px;
  line-height: 1.5;

  color: #777777;
}

.extension {
  margin-top: 16px;

  font-size: 20px;
  line-height: 1.6;

  color: #777777;
}

.conference-link {
  color: #4f6fae;
  text-decoration: none;
}

.conference-link:hover {
  text-decoration: underline;
}

.resource-links {
  display: flex;
  justify-content: center;

  margin-top: 22px;
}

.resource-button {
  display: inline-flex;
  align-items: center;

  gap: 8px;

  padding: 10px 18px;

  color: #315f9f;
  background: #edf3fd;

  border: none;
  border-radius: 999px;

  font-size: 16px;
  line-height: 1;
  font-weight: 600;

  text-decoration: none;

  transition:
    color 0.16s ease,
    background 0.16s ease,
    box-shadow 0.16s ease,
    transform 0.16s ease;
}

.resource-button svg {
  width: 18px;
  height: 18px;

  fill: currentColor;
}

.resource-button:hover {
  color: #2457a0;
  background: #e3edfc;

  box-shadow: 0 6px 16px rgba(50, 115, 220, 0.16);

  transform: translateY(-1px);
}
</style>

