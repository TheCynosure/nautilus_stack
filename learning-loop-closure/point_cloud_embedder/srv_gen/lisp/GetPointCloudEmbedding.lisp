; Auto-generated. Do not edit!


(cl:in-package point_cloud_embedder-srv)


;//! \htmlinclude GetPointCloudEmbedding-request.msg.html

(cl:defclass <GetPointCloudEmbedding-request> (roslisp-msg-protocol:ros-message)
  ((cloud
    :reader cloud
    :initarg :cloud
    :type sensor_msgs-msg:PointCloud2
    :initform (cl:make-instance 'sensor_msgs-msg:PointCloud2)))
)

(cl:defclass GetPointCloudEmbedding-request (<GetPointCloudEmbedding-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetPointCloudEmbedding-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetPointCloudEmbedding-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name point_cloud_embedder-srv:<GetPointCloudEmbedding-request> is deprecated: use point_cloud_embedder-srv:GetPointCloudEmbedding-request instead.")))

(cl:ensure-generic-function 'cloud-val :lambda-list '(m))
(cl:defmethod cloud-val ((m <GetPointCloudEmbedding-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader point_cloud_embedder-srv:cloud-val is deprecated.  Use point_cloud_embedder-srv:cloud instead.")
  (cloud m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetPointCloudEmbedding-request>) ostream)
  "Serializes a message object of type '<GetPointCloudEmbedding-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'cloud) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetPointCloudEmbedding-request>) istream)
  "Deserializes a message object of type '<GetPointCloudEmbedding-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'cloud) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetPointCloudEmbedding-request>)))
  "Returns string type for a service object of type '<GetPointCloudEmbedding-request>"
  "point_cloud_embedder/GetPointCloudEmbeddingRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetPointCloudEmbedding-request)))
  "Returns string type for a service object of type 'GetPointCloudEmbedding-request"
  "point_cloud_embedder/GetPointCloudEmbeddingRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetPointCloudEmbedding-request>)))
  "Returns md5sum for a message object of type '<GetPointCloudEmbedding-request>"
  "a2c07223ab3a99859d0995f0e704a95c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetPointCloudEmbedding-request)))
  "Returns md5sum for a message object of type 'GetPointCloudEmbedding-request"
  "a2c07223ab3a99859d0995f0e704a95c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetPointCloudEmbedding-request>)))
  "Returns full string definition for message of type '<GetPointCloudEmbedding-request>"
  (cl:format cl:nil "sensor_msgs/PointCloud2 cloud~%~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetPointCloudEmbedding-request)))
  "Returns full string definition for message of type 'GetPointCloudEmbedding-request"
  (cl:format cl:nil "sensor_msgs/PointCloud2 cloud~%~%================================================================================~%MSG: sensor_msgs/PointCloud2~%# This message holds a collection of N-dimensional points, which may~%# contain additional information such as normals, intensity, etc. The~%# point data is stored as a binary blob, its layout described by the~%# contents of the \"fields\" array.~%~%# The point cloud data may be organized 2d (image-like) or 1d~%# (unordered). Point clouds organized as 2d images may be produced by~%# camera depth sensors such as stereo or time-of-flight.~%~%# Time of sensor data acquisition, and the coordinate frame ID (for 3d~%# points).~%Header header~%~%# 2D structure of the point cloud. If the cloud is unordered, height is~%# 1 and width is the length of the point cloud.~%uint32 height~%uint32 width~%~%# Describes the channels and their layout in the binary data blob.~%PointField[] fields~%~%bool    is_bigendian # Is this data bigendian?~%uint32  point_step   # Length of a point in bytes~%uint32  row_step     # Length of a row in bytes~%uint8[] data         # Actual point data, size is (row_step*height)~%~%bool is_dense        # True if there are no invalid points~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/PointField~%# This message holds the description of one point entry in the~%# PointCloud2 message format.~%uint8 INT8    = 1~%uint8 UINT8   = 2~%uint8 INT16   = 3~%uint8 UINT16  = 4~%uint8 INT32   = 5~%uint8 UINT32  = 6~%uint8 FLOAT32 = 7~%uint8 FLOAT64 = 8~%~%string name      # Name of field~%uint32 offset    # Offset from start of point struct~%uint8  datatype  # Datatype enumeration, see above~%uint32 count     # How many elements in the field~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetPointCloudEmbedding-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'cloud))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetPointCloudEmbedding-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetPointCloudEmbedding-request
    (cl:cons ':cloud (cloud msg))
))
;//! \htmlinclude GetPointCloudEmbedding-response.msg.html

(cl:defclass <GetPointCloudEmbedding-response> (roslisp-msg-protocol:ros-message)
  ((embedding
    :reader embedding
    :initarg :embedding
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass GetPointCloudEmbedding-response (<GetPointCloudEmbedding-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetPointCloudEmbedding-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetPointCloudEmbedding-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name point_cloud_embedder-srv:<GetPointCloudEmbedding-response> is deprecated: use point_cloud_embedder-srv:GetPointCloudEmbedding-response instead.")))

(cl:ensure-generic-function 'embedding-val :lambda-list '(m))
(cl:defmethod embedding-val ((m <GetPointCloudEmbedding-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader point_cloud_embedder-srv:embedding-val is deprecated.  Use point_cloud_embedder-srv:embedding instead.")
  (embedding m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetPointCloudEmbedding-response>) ostream)
  "Serializes a message object of type '<GetPointCloudEmbedding-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'embedding))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'embedding))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetPointCloudEmbedding-response>) istream)
  "Deserializes a message object of type '<GetPointCloudEmbedding-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'embedding) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'embedding)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetPointCloudEmbedding-response>)))
  "Returns string type for a service object of type '<GetPointCloudEmbedding-response>"
  "point_cloud_embedder/GetPointCloudEmbeddingResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetPointCloudEmbedding-response)))
  "Returns string type for a service object of type 'GetPointCloudEmbedding-response"
  "point_cloud_embedder/GetPointCloudEmbeddingResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetPointCloudEmbedding-response>)))
  "Returns md5sum for a message object of type '<GetPointCloudEmbedding-response>"
  "a2c07223ab3a99859d0995f0e704a95c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetPointCloudEmbedding-response)))
  "Returns md5sum for a message object of type 'GetPointCloudEmbedding-response"
  "a2c07223ab3a99859d0995f0e704a95c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetPointCloudEmbedding-response>)))
  "Returns full string definition for message of type '<GetPointCloudEmbedding-response>"
  (cl:format cl:nil "float32[] embedding~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetPointCloudEmbedding-response)))
  "Returns full string definition for message of type 'GetPointCloudEmbedding-response"
  (cl:format cl:nil "float32[] embedding~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetPointCloudEmbedding-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'embedding) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetPointCloudEmbedding-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetPointCloudEmbedding-response
    (cl:cons ':embedding (embedding msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetPointCloudEmbedding)))
  'GetPointCloudEmbedding-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetPointCloudEmbedding)))
  'GetPointCloudEmbedding-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetPointCloudEmbedding)))
  "Returns string type for a service object of type '<GetPointCloudEmbedding>"
  "point_cloud_embedder/GetPointCloudEmbedding")