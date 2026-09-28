# Project website and search visibility

The public project page is [qiaoyuan-zheng.com/POLAR-Bench](https://qiaoyuan-zheng.com/POLAR-Bench/).
It is a static GitHub Pages site published from **`main` → `/docs`**.
The account already uses `qiaoyuan-zheng.com` as its personal Pages domain.
GitHub automatically redirects `https://qiaoyuan667.github.io/POLAR-Bench/` to
the project path on that domain; this project does not change the personal site
or its DNS settings. Canonical metadata and the sitemap use the final URL.
No package installation, JavaScript framework, account signup, or paid server is
needed to view the site. `.nojekyll` keeps the supplied HTML and assets intact.

## Edit and publish

- `index.html`: visible article content, resource links, canonical URL, citation
  metadata, and Schema.org `WebSite`, `ScholarlyArticle`, and `Dataset` records.
- `site-assets/site.css`: responsive page styles.
- `site-assets/site.js`: optional copy-to-clipboard enhancement. Reading and
  navigation do not depend on JavaScript.
- `site-assets/`: original paper images copied from `../assets/figures/` so the
  published `/docs` directory is self-contained. Update both copies together.
- `polar-bench.bib`: downloadable citation; keep it consistent with the visible
  BibTeX and repository README.
- `sitemap.xml`: canonical page URL and truthful last-modified date. Update its
  date when the page materially changes.

Changes pushed to `main` trigger Pages publication. Keep the publication source
limited to `/docs`; do not publish credentials, result transcripts, development
environments, or the full repository as a web bundle. Current documentation
Markdown files are already public in the repository and remain available there.

All page text is present in the initial HTML. The site uses system fonts, local
paper images, and no analytics, tracking, external JavaScript, or runtime API.
This page's layout was independently implemented with an academic editorial
presentation inspired by the reference project; no third-party template code
or figures were copied.

## What is configured for search

The homepage includes a descriptive title and summary, one H1 naming POLAR-Bench,
a canonical URL, crawlable resource links, image descriptions, citation metadata,
and structured paper/dataset records. It has no login wall or `noindex` directive.
The sitemap is at:

<https://qiaoyuan-zheng.com/POLAR-Bench/sitemap.xml>

**Do not add a project-directory `robots.txt` and assume it controls crawling.**
Google reads robots rules from the host root, which for this site is
`https://qiaoyuan-zheng.com/robots.txt`. A file under `/POLAR-Bench/` does not
control the host. Do not edit the account's root website merely to change this
project's crawl settings.

## Optional Google Search Console submission

1. In [Google Search Console](https://search.google.com/search-console), add the
   URL-prefix property `https://qiaoyuan-zheng.com/POLAR-Bench/`.
2. Complete Google's ownership verification. For an HTML tag/file method, add
   the exact tag or file Google supplies to this site's publishing directory;
   do not invent a verification value.
3. Submit `sitemap.xml` under Sitemaps.
4. Inspect the homepage URL and choose **Request indexing** if available.

This is an owner action until ownership has been verified. Merely publishing a
page or adding metadata does not submit it to Search Console. Google decides
whether and when to crawl, index, and rank it; neither a sitemap nor a request
guarantees inclusion or a particular position for “POLAR-Bench”.

Official guidance: [Google search fundamentals](https://developers.google.com/search/docs/fundamentals/get-started-developers),
[sitemaps](https://developers.google.com/search/docs/crawling-indexing/sitemaps/overview),
and [requesting recrawling](https://developers.google.com/search/docs/crawling-indexing/ask-google-to-recrawl).
