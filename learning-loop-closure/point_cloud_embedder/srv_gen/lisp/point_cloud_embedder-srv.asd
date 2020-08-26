
(cl:in-package :asdf)

(defsystem "point_cloud_embedder-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "GetPointCloudEmbedding" :depends-on ("_package_GetPointCloudEmbedding"))
    (:file "_package_GetPointCloudEmbedding" :depends-on ("_package"))
  ))