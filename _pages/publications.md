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



