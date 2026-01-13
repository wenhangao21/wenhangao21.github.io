---
layout: archive
title: "First or Corresponding Author Publications"
permalink: /publications/
author_profile: true
---
\* indicates equal contributors/co-first authors; ‡ indicates equal senior contributors/co-corresponding authors

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

# First or Co-first Author Preprints
* One paper on LLM-assisted drug editing; Submitted to ACL 2026; **W.Gao** el al.
* One paper on size-consistent diffusion models for 3D molecular generation; Submitted to ICLR 2026; **W.Gao**, J.Qu, Y.Liu
* One paper that reveals neural operators can learn hidden physics from data; In Submission to ICML 2026; **W.Gao\***, J.Luo\*, R.Xu, F.Wan, X.Liu, Y.Liu

