<style>
/* 1) Hide footer + remove its spacing */
.page__footer,
.page__footer-follow,
.page__footer-copyright {
  display: none !important;
  margin: 0 !important;
  padding: 0 !important;
  height: 0 !important;
}

/* 2) Remove theme bottom padding that can create "white space" */
#main,
.initial-content,
.page,
.page__inner-wrap,
.page__content,
.archive {
  margin: 0 !important;
  padding: 0 !important;
}

/* 3) Full-viewport iframe */
.fullscreen-embed {
  position: fixed;
  inset: 0;
  z-index: 0;              /* keep it normal (NOT -1) */
}
.fullscreen-embed iframe {
  width: 100vw;
  height: 100vh;
  border: 0;
  display: block;
}

/* 4) Put masthead/nav above the iframe (this is the key) */
.masthead,
.greedy-nav,
.masthead__inner-wrap {
  position: relative !important;   /* make z-index apply */
  z-index: 9999 !important;
}
</style>

<div class="fullscreen-embed">
  <iframe src="https://wenhanacademia-ai-paper-finder.hf.space"></iframe>
</div>
