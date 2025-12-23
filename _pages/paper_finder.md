---
permalink: /paper_finder/
title: "AI Conference Paper Finder"
author_profile: false
redirect_from:
  - /md/
  - /markdown.html
---

<style>
/* Hide footer */
.page__footer,
.page__footer-follow,
.page__footer-copyright{
  display:none !important;
  margin:0 !important;
  padding:0 !important;
  height:0 !important;
}

/* Remove extra padding/margins from the theme wrappers */
#main,
.initial-content,
.page,
.page__inner-wrap,
.page__content,
.archive{
  margin:0 !important;
  padding:0 !important;
}

/* Keep nav above iframe */
.masthead,
.masthead__inner-wrap,
.greedy-nav{
  position: relative !important;
  z-index: 9999 !important;
}

/* ---- KEY PART: iframe starts below header ---- */
:root { --mh: 70px; } /* fallback */

.fullscreen-embed{
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: var(--mh);              /* offset by masthead height */
  z-index: 0;
}
.fullscreen-embed iframe{
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
}

/* Optional: prevent the page itself from scrolling behind the iframe */
html, body { height: 100%; }
body { overflow: hidden; }
</style>

<div class="fullscreen-embed">
  <iframe src="https://wenhanacademia-ai-paper-finder.hf.space"></iframe>
</div>

<script>
(function () {
  function setMastheadHeight() {
    var mh = document.querySelector('.masthead');
    var h = mh ? mh.offsetHeight : 0;
    document.documentElement.style.setProperty('--mh', h + 'px');
  }
  window.addEventListener('load', setMastheadHeight);
  window.addEventListener('resize', setMastheadHeight);
  setMastheadHeight();
})();
</script>
