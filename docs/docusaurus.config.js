module.exports = {
  title: 'Nautilus',
  tagline: 'A Map Curation Tool with Autonomous Loop Closure',
  url: 'https://thecynosure.github.io',
  baseUrl: '/nautilus_docs/',
  favicon: 'img/favicon.ico',
  organizationName: 'TheCynosure', // Usually your GitHub org/user name.
  projectName: 'nautilus_docs', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'Nautilus',
      logo: {
        alt: 'My Site Logo',
        src: 'img/logo.svg',
      },
      links: [
        {
          to: 'docs/getting_started',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Style Guide',
              to: 'https://google.github.io/styleguide/cppguide.html',
            },
            {
              label: 'Getting Started',
              to: 'docs/getting_started',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/ut-amrl/nautilus',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Nautilus, Inc. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/master/website/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/master/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
