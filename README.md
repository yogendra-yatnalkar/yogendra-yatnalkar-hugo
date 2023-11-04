# Hugo-files

- The **public** folder contains https://github.com/yogendra-yatnalkar/yogendra-yatnalkar.github.io repository as git-submodule.

### Updates to origin hugo-book code:

- Comments using `utterences`
  - Enabled with file: `{theme_name}/layouts/partial/docs/comments.html`
- All images displayed will be centered using: `{theme_name}/assets/_custom.scss`
- Enabled ugly urls (Enable using `config.toml` file)
  - uglyurls = true
  - Advantage: markdown files are able to render images using relative paths
- Open web-links in new-tab: (https://discourse.gohugo.io/t/simple-way-to-open-in-a-new-tab/28677/4)
  - Updated code in: `layouts/_default/_markup/render-link.html`
- Added pdf support for resume using: https://github.com/anvithks/hugo-embed-pdf-shortcode/ 
  - for adding pdf's, uglyurls are causing an issue. Hence in the frontmatter of individual markdown file, change to url to not display ".html" extension.
  - Refer to the markdown of "Resume" section
- Added Google Analytics for user tracking. The google-analytics keyword needs to be placed on the top line of the "config.toml" file. 

Notes:
  - The folder names where content is placed should have lower-case characters only
  - I believe the search space in hugo does not index the first "_index.md" file present in the /content directory



