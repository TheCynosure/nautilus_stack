
import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  
{
  path: '/nautilus_docs/',
  component: ComponentCreator('/nautilus_docs/'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog',
  component: ComponentCreator('/nautilus_docs/blog'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/hello-world',
  component: ComponentCreator('/nautilus_docs/blog/hello-world'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/hola',
  component: ComponentCreator('/nautilus_docs/blog/hola'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/tags',
  component: ComponentCreator('/nautilus_docs/blog/tags'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/tags/docusaurus',
  component: ComponentCreator('/nautilus_docs/blog/tags/docusaurus'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/tags/facebook',
  component: ComponentCreator('/nautilus_docs/blog/tags/facebook'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/tags/hello',
  component: ComponentCreator('/nautilus_docs/blog/tags/hello'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/tags/hola',
  component: ComponentCreator('/nautilus_docs/blog/tags/hola'),
  exact: true,
  
},
{
  path: '/nautilus_docs/blog/welcome',
  component: ComponentCreator('/nautilus_docs/blog/welcome'),
  exact: true,
  
},
{
  path: '/nautilus_docs/docs/:route',
  component: ComponentCreator('/nautilus_docs/docs/:route'),
  
  routes: [
{
  path: '/nautilus_docs/docs/config_params',
  component: ComponentCreator('/nautilus_docs/docs/config_params'),
  exact: true,
  
},
{
  path: '/nautilus_docs/docs/getting_started',
  component: ComponentCreator('/nautilus_docs/docs/getting_started'),
  exact: true,
  
},
{
  path: '/nautilus_docs/docs/install_nautilus',
  component: ComponentCreator('/nautilus_docs/docs/install_nautilus'),
  exact: true,
  
},
{
  path: '/nautilus_docs/docs/run_nautilus',
  component: ComponentCreator('/nautilus_docs/docs/run_nautilus'),
  exact: true,
  
},
{
  path: '/nautilus_docs/docs/running_example_bags',
  component: ComponentCreator('/nautilus_docs/docs/running_example_bags'),
  exact: true,
  
},
{
  path: '/nautilus_docs/docs/vectorize',
  component: ComponentCreator('/nautilus_docs/docs/vectorize'),
  exact: true,
  
},
{
  path: '/nautilus_docs/docs/write_config',
  component: ComponentCreator('/nautilus_docs/docs/write_config'),
  exact: true,
  
}],
},
  
  {
    path: '*',
    component: ComponentCreator('*')
  }
];
