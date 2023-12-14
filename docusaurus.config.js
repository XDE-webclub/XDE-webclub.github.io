// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'xde-web',
  tagline: 'xde-web',
  favicon: 'img/github.svg',

  // Set the production url of your site here
  url: 'https://xde-webclub.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'jiangmiemie', // Usually your GitHub org/user name.
  projectName: 'DocusaurusBlog', // Usually your repo name.
  deploymentBranch: 'gh-pages',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
    path: 'i18n',
    localeConfigs: {
      zh: {
        label: '中文',
        direction: 'ltr',
        htmlLang: '	zh-Hans',
        calendar: 'gregory',
        path: 'zh-Hans',
      },

    },
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: 'docs',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/jiangmiemie/blog/blob/master',
        },

        blog: {
          path: "blog",
          routeBasePath: "blog",
          showReadingTime: true,
          postsPerPage: 'ALL',
          blogSidebarCount: 'ALL',
          // editUrl:
          //   'https://github.com/jiangmiemie/blog/blob/master',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/github.svg',

      navbar: {
        title: '首页',
        hideOnScroll: true,

        items: [
          // {
          //   type: 'docSidebar',
          //   sidebarId: 'tutorialSidebar',
          //   position: 'left',
          //   label: 'Start',
          // },
          {
            to: "/docs",
            position: 'left',
            label: 'Python',
          },

          {
            to: "/blog",
            position: 'left',
            label: '博客',
          },

        ],
      },

      footer: {
        style: 'light',
        copyright: `控江XDE-web`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        defaultLanguage: "markdown",
      },

    }),
};

module.exports = config;
