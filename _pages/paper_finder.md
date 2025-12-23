---
permalink: /paper_finder/
title: "AI Conference Paper Finder"
author_profile: false
redirect_from: 
  - /md/
  - /markdown.html
---
<style>
/* Hide footer (and any subparts) */
.page__footer,
.page__footer-follow,
.page__footer-copyright {
  display: none !important;
  margin: 0 !important;
  padding: 0 !important;
  height: 0 !important;
}

/* Nuke all bottom spacing the theme might add */
#main,
.initial-content,
.page,
.page__inner-wrap,
.page__content,
.archive {
  margin: 0 !important;
  padding: 0 !important;
}

/* Optional: if you want ONLY the iframe (no title/meta spacing) */
.page__title,
.page__meta,
.page__share {
  display: none !important;
}

/* Make iframe fill viewport */
.fullscreen-embed {
  position: fixed;
  inset: 0;          /* top/right/bottom/left = 0 */
  z-index: 0;
}
.fullscreen-embed iframe {
  width: 100vw;
  height: 100vh;
  border: 0;
  display: block;
}

/* No scrolling because the page is now just the viewport */
html, body { height: 100%; }
body { overflow: hidden; }
</style>

<div class="fullscreen-embed">
  <iframe src="https://wenhanacademia-ai-paper-finder.hf.space"></iframe>
</div>
