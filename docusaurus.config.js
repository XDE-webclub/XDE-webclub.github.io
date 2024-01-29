// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
// 导入公式渲染模块remark-math和rehype-katex
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// 设置明暗模式
const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

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
  organizationName: 'XDE-web', // Usually your GitHub org/user name.
  projectName: 'DocusaurusBlog', // Usually your repo name.
  deploymentBranch: 'gh-pages',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: 'docs',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          sidebarPath: require.resolve('./sidebars.js'),
        },

        blog: false,
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
  //用于与katex配合获得更好的公式渲染效果
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
};

module.exports = config;
