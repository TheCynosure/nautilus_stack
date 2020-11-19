export default {
  "plugins": [],
  "themes": [],
  "customFields": {},
  "themeConfig": {
    "navbar": {
      "title": "Nautilus",
      "logo": {
        "alt": "My Site Logo",
        "src": "img/logo.svg"
      },
      "links": [
        {
          "to": "docs/getting_started",
          "activeBasePath": "docs",
          "label": "Docs",
          "position": "left"
        }
      ]
    },
    "footer": {
      "style": "dark",
      "links": [
        {
          "title": "Docs",
          "items": [
            {
              "label": "Style Guide",
              "to": "https://google.github.io/styleguide/cppguide.html"
            },
            {
              "label": "Getting Started",
              "to": "docs/getting_started"
            }
          ]
        },
        {
          "title": "More",
          "items": [
            {
              "label": "GitHub",
              "href": "https://github.com/ut-amrl/nautilus"
            }
          ]
        }
      ],
      "copyright": "Copyright © 2020 Nautilus, Inc. Built with Docusaurus."
    }
  },
  "title": "Nautilus",
  "tagline": "A Map Curation Tool with Autonomous Loop Closure",
  "url": "https://thecynosure.github.io",
  "baseUrl": "/nautilus_docs/",
  "favicon": "img/favicon.ico",
  "organizationName": "TheCynosure",
  "projectName": "nautilus_docs",
  "presets": [
    [
      "@docusaurus/preset-classic",
      {
        "docs": {
          "sidebarPath": "/home/jack/catkin_ws/src/nautilus_stack/docs/sidebars.js",
          "editUrl": "https://github.com/facebook/docusaurus/edit/master/website/"
        },
        "blog": {
          "showReadingTime": true,
          "editUrl": "https://github.com/facebook/docusaurus/edit/master/website/blog/"
        },
        "theme": {
          "customCss": "/home/jack/catkin_ws/src/nautilus_stack/docs/src/css/custom.css"
        }
      }
    ]
  ]
};