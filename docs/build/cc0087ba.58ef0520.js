(window.webpackJsonp=window.webpackJsonp||[]).push([[24],{161:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return i})),n.d(t,"metadata",(function(){return a})),n.d(t,"rightToc",(function(){return c})),n.d(t,"default",(function(){return u}));var o=n(2),r=n(9),l=(n(0),n(170)),i={id:"config_params",title:"Config Parameters"},a={id:"config_params",title:"Config Parameters",description:"### General",source:"@site/docs/config_params.mdx",permalink:"/nautilus_docs/docs/config_params",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/docs/config_params.mdx",sidebar:"sidebar",previous:{title:"Write your own Config",permalink:"/nautilus_docs/docs/write_config"}},c=[{value:"General",id:"general",children:[]},{value:"Performance",id:"performance",children:[]},{value:"Human-In-The-Loop",id:"human-in-the-loop",children:[]},{value:"Autonomous Loop Closure",id:"autonomous-loop-closure",children:[]},{value:"Normal Computation",id:"normal-computation",children:[]}],b={rightToc:c};function u(e){var t=e.components,n=Object(r.a)(e,["components"]);return Object(l.b)("wrapper",Object(o.a)({},b,n,{components:t,mdxType:"MDXLayout"}),Object(l.b)("h3",{id:"general"},"General"),Object(l.b)("h4",{id:"bag_path"},"bag_path"),Object(l.b)("p",null,"Type: String"),Object(l.b)("p",null,"The path from the root of the project to the ROS bag file to be processed."),Object(l.b)("h4",{id:"pose_number"},"pose_number"),Object(l.b)("p",null,"Type: Integer"),Object(l.b)("p",null,"The number of lidar scans (and poses) to load from the bag file."),Object(l.b)("h4",{id:"odom_topic"},"odom_topic"),Object(l.b)("p",null,"Type: String"),Object(l.b)("p",null,"The topic to pull odometry data from, this topic should have the ",Object(l.b)("inlineCode",{parentName:"p"},"nav_msgs/Odometry")," data type, or if differential_odom is set to true, it can be of type ",Object(l.b)("inlineCode",{parentName:"p"},"CobotMsgs/CobotOdometry"),"."),Object(l.b)("h4",{id:"lidar_topic"},"lidar_topic"),Object(l.b)("p",null,"Type: String"),Object(l.b)("p",null,"The topic to pull LiDAR data from, this topic should have the ",Object(l.b)("inlineCode",{parentName:"p"},"sensor_msgs/LaserScan")," type."),Object(l.b)("h4",{id:"differential_odom"},"differential_odom"),Object(l.b)("p",null,"Type: Boolean"),Object(l.b)("p",null,"True if using CobotOdometry message, false otherwise."),Object(l.b)("h4",{id:"pose_output_file"},"pose_output_file"),Object(l.b)("p",null,"Type: String"),Object(l.b)("p",null,"The path to the file to output the pose positions too, this writing will be triggered when the user hits the WriteButton in RViz."),Object(l.b)("h4",{id:"map_output_file"},"map_output_file"),Object(l.b)("p",null,"Type: String"),Object(l.b)("p",null,"The path to output the text file to output the vector map to. If unset then when the user hits the Vectorize Button in RViz there will be no text output."),Object(l.b)("h3",{id:"performance"},"Performance"),Object(l.b)("h4",{id:"translation_weight"},"translation_weight"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"The multiplier applied to the translation error when optimizing the pose-graph."),Object(l.b)("h4",{id:"rotation_weight"},"rotation_weight"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"The multiplier applied to the rotation error when optimizing the pose-graph."),Object(l.b)("h4",{id:"stopping_accuracy"},"stopping_accuracy"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"When the scans change by less than this amount (in meters), the optimization iteration is deemed solved."),Object(l.b)("h4",{id:"max_lidar_range"},"max_lidar_range"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"The maximum distance LiDAR data is allowed to be from the robot."),Object(l.b)("h4",{id:"rotation_change_for_lidar"},"rotation_change_for_lidar"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"The amount of rotation needed to signal a lidar scan capture (in radians)."),Object(l.b)("h4",{id:"translation_change_for_lidar"},"translation_change_for_lidar"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"The amount of translation needed to signal a lidar scan capture."),Object(l.b)("h4",{id:"lidar_constraint_amount"},"lidar_constraint_amount"),Object(l.b)("p",null,"Type: Integer"),Object(l.b)("p",null,"The number of poses to compare each pose too. For example the default is 10, therefore we compare and try to minimize error between the current pose and the past ten, for every pose."),Object(l.b)("h4",{id:"outlier_threshold"},"outlier_threshold"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"Points further than this distance (in meters) cannot be counted as the same point during ICL/ICP in the optimization phase."),Object(l.b)("h3",{id:"human-in-the-loop"},"Human-In-The-Loop"),Object(l.b)("h4",{id:"hitl_lc_topic"},"hitl_lc_topic"),Object(l.b)("p",null,"Type: String"),Object(l.b)("p",null,"The topic that HITL Slam tool send the message to trigger HITL Slam, ",Object(l.b)("strong",{parentName:"p"},"probably best to not change unless you know what you are doing"),"."),Object(l.b)("h4",{id:"hitl_line_width"},"hitl_line_width"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"Points further than this will not count as falling on the HITL LC line and therefore will not be used for HITLSlam."),Object(l.b)("h4",{id:"hitl_pose_point_threshold"},"hitl_pose_point_threshold"),Object(l.b)("p",null,"Type: Integer"),Object(l.b)("p",null,"The amount of points needed to include a pose for HITL Slam."),Object(l.b)("h3",{id:"autonomous-loop-closure"},"Autonomous Loop Closure"),Object(l.b)("h4",{id:"auto_lc"},"auto_lc"),Object(l.b)("p",null,"Type: Boolean"),Object(l.b)("p",null,"Automatically loop close or not"),Object(l.b)("h4",{id:"keyframe_chi_squared_test"},"keyframe_chi_squared_test"),Object(l.b)("p",null,"Type: Boolean"),Object(l.b)("p",null,"Whether or not to use chi_squared test for keyframes"),Object(l.b)("h4",{id:"keyframe_min_odom_distance"},"keyframe_min_odom_distance"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"Distance between keyframes if chi^2 is not in use"),Object(l.b)("h4",{id:"keyframe_local_uncertainty_filtering"},"keyframe_local_uncertainty_filtering"),Object(l.b)("p",null,"Type: Boolean"),Object(l.b)("p",null,"Whether or not to use local uncertainty filtering for keyframes"),Object(l.b)("h4",{id:"local_uncertainty_condition_threshold"},"local_uncertainty_condition_threshold"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"All scans with local uncertainty less than this threshold are one step closer to being used for automatic lc, if keyframe_local_uncertainty_filtering is on"),Object(l.b)("h4",{id:"local_uncertainty_scale_threshold"},"local_uncertainty_scale_threshold"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"All scans with local uncertainty scale less than this threshold are one step closer to being used for automatic lc, if keyframe_local_uncertainty_filtering is on"),Object(l.b)("h4",{id:"local_uncertainty_prev_scans"},"local_uncertainty_prev_scans"),Object(l.b)("p",null,"Type: Integer"),Object(l.b)("p",null,"The amount of previous scans to use for calculating local uncertainty  if keyframe_local_uncertainty_filter    ing is on"),Object(l.b)("h4",{id:"lc_match_threshold"},"lc_match_threshold"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"Threshold used in automatic loop closure."),Object(l.b)("h4",{id:"lc_base_max_range"},"lc_base_max_range"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"Base max range to consider a loop closure"),Object(l.b)("h4",{id:"lc_max_range_scaling"},"lc_max_range_scaling"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"How much max range to consider a loop closure increases as nodes get more distant"),Object(l.b)("h4",{id:"lc_translation_weight"},"lc_translation_weight"),Object(l.b)("p",null,"Type: Integer"),Object(l.b)("p",null,"Loop closure translation multiplier, used during loop closure for odometry residuals."),Object(l.b)("h4",{id:"lc_rotation_weight"},"lc_rotation_weight"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"Loop closure rotation multiplier, used during loop closure for odometry residuals."),Object(l.b)("h4",{id:"lc_min_keyframes"},"lc_min_keyframes"),Object(l.b)("p",null,"Type: Integer"),Object(l.b)("p",null,"Minimum number of keyframes that must exist between loop closures."),Object(l.b)("h4",{id:"lc_debug_output_dir"},"lc_debug_output_dir"),Object(l.b)("p",null,"Type: String"),Object(l.b)("p",null,"Used to dump images from auto-lc"),Object(l.b)("h3",{id:"normal-computation"},"Normal Computation"),Object(l.b)("h4",{id:"nc_neighborhood_size"},"nc_neighborhood_size"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"The neighborhood size to consider when RANSACing for normals."),Object(l.b)("h4",{id:"nc_neighborhood_step_size"},"nc_neighborhood_step_size"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"How much the neighborhood increases with each iteration."),Object(l.b)("h4",{id:"nc_mean_distance"},"nc_mean_distance"),Object(l.b)("p",null,"Type: Double"),Object(l.b)("p",null,"You got me, this one's just a constant."),Object(l.b)("h4",{id:"nc_bin_number"},"nc_bin_number"),Object(l.b)("p",null,"Type: Integer"),Object(l.b)("p",null,"The number of buckets to use in the Hough transform."))}u.isMDXComponent=!0},170:function(e,t,n){"use strict";n.d(t,"a",(function(){return p})),n.d(t,"b",(function(){return d}));var o=n(0),r=n.n(o);function l(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){l(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,o,r=function(e,t){if(null==e)return{};var n,o,r={},l=Object.keys(e);for(o=0;o<l.length;o++)n=l[o],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(o=0;o<l.length;o++)n=l[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var b=r.a.createContext({}),u=function(e){var t=r.a.useContext(b),n=t;return e&&(n="function"==typeof e?e(t):a({},t,{},e)),n},p=function(e){var t=u(e.components);return r.a.createElement(b.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},h=Object(o.forwardRef)((function(e,t){var n=e.components,o=e.mdxType,l=e.originalType,i=e.parentName,b=c(e,["components","mdxType","originalType","parentName"]),p=u(n),h=o,d=p["".concat(i,".").concat(h)]||p[h]||s[h]||l;return n?r.a.createElement(d,a({ref:t},b,{components:n})):r.a.createElement(d,a({ref:t},b))}));function d(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var l=n.length,i=new Array(l);i[0]=h;var a={};for(var c in t)hasOwnProperty.call(t,c)&&(a[c]=t[c]);a.originalType=e,a.mdxType="string"==typeof e?e:o,i[1]=a;for(var b=2;b<l;b++)i[b]=n[b];return r.a.createElement.apply(null,i)}return r.a.createElement.apply(null,n)}h.displayName="MDXCreateElement"}}]);