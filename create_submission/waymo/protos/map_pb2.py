# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: waymo_open_dataset/protos/map.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='waymo_open_dataset/protos/map.proto',
  package='waymo.open_dataset',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n#waymo_open_dataset/protos/map.proto\x12\x12waymo.open_dataset\"u\n\x03Map\x12\x34\n\x0cmap_features\x18\x01 \x03(\x0b\x32\x1e.waymo.open_dataset.MapFeature\x12\x38\n\x0e\x64ynamic_states\x18\x02 \x03(\x0b\x32 .waymo.open_dataset.DynamicState\"j\n\x0c\x44ynamicState\x12\x19\n\x11timestamp_seconds\x18\x01 \x01(\x01\x12?\n\x0blane_states\x18\x02 \x03(\x0b\x32*.waymo.open_dataset.TrafficSignalLaneState\"\x8c\x03\n\x16TrafficSignalLaneState\x12\x0c\n\x04lane\x18\x01 \x01(\x03\x12?\n\x05state\x18\x02 \x01(\x0e\x32\x30.waymo.open_dataset.TrafficSignalLaneState.State\x12\x30\n\nstop_point\x18\x03 \x01(\x0b\x32\x1c.waymo.open_dataset.MapPoint\"\xf0\x01\n\x05State\x12\x16\n\x12LANE_STATE_UNKNOWN\x10\x00\x12\x19\n\x15LANE_STATE_ARROW_STOP\x10\x01\x12\x1c\n\x18LANE_STATE_ARROW_CAUTION\x10\x02\x12\x17\n\x13LANE_STATE_ARROW_GO\x10\x03\x12\x13\n\x0fLANE_STATE_STOP\x10\x04\x12\x16\n\x12LANE_STATE_CAUTION\x10\x05\x12\x11\n\rLANE_STATE_GO\x10\x06\x12\x1c\n\x18LANE_STATE_FLASHING_STOP\x10\x07\x12\x1f\n\x1bLANE_STATE_FLASHING_CAUTION\x10\x08\"\xda\x02\n\nMapFeature\x12\n\n\x02id\x18\x01 \x01(\x03\x12.\n\x04lane\x18\x03 \x01(\x0b\x32\x1e.waymo.open_dataset.LaneCenterH\x00\x12\x31\n\troad_line\x18\x04 \x01(\x0b\x32\x1c.waymo.open_dataset.RoadLineH\x00\x12\x31\n\troad_edge\x18\x05 \x01(\x0b\x32\x1c.waymo.open_dataset.RoadEdgeH\x00\x12\x31\n\tstop_sign\x18\x07 \x01(\x0b\x32\x1c.waymo.open_dataset.StopSignH\x00\x12\x32\n\tcrosswalk\x18\x08 \x01(\x0b\x32\x1d.waymo.open_dataset.CrosswalkH\x00\x12\x33\n\nspeed_bump\x18\t \x01(\x0b\x32\x1d.waymo.open_dataset.SpeedBumpH\x00\x42\x0e\n\x0c\x66\x65\x61ture_data\"+\n\x08MapPoint\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\"\x82\x02\n\nLaneCenter\x12\x17\n\x0fspeed_limit_mph\x18\x01 \x01(\x01\x12\x35\n\x04type\x18\x02 \x01(\x0e\x32\'.waymo.open_dataset.LaneCenter.LaneType\x12\x15\n\rinterpolating\x18\x03 \x01(\x08\x12.\n\x08polyline\x18\x08 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\"]\n\x08LaneType\x12\x12\n\x0eTYPE_UNDEFINED\x10\x00\x12\x10\n\x0cTYPE_FREEWAY\x10\x01\x12\x17\n\x13TYPE_SURFACE_STREET\x10\x02\x12\x12\n\x0eTYPE_BIKE_LANE\x10\x03\"\xcd\x01\n\x08RoadEdge\x12\x37\n\x04type\x18\x01 \x01(\x0e\x32).waymo.open_dataset.RoadEdge.RoadEdgeType\x12.\n\x08polyline\x18\x02 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\"X\n\x0cRoadEdgeType\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x1b\n\x17TYPE_ROAD_EDGE_BOUNDARY\x10\x01\x12\x19\n\x15TYPE_ROAD_EDGE_MEDIAN\x10\x02\"\x88\x03\n\x08RoadLine\x12\x37\n\x04type\x18\x01 \x01(\x0e\x32).waymo.open_dataset.RoadLine.RoadLineType\x12.\n\x08polyline\x18\x02 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\"\x92\x02\n\x0cRoadLineType\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x1c\n\x18TYPE_BROKEN_SINGLE_WHITE\x10\x01\x12\x1b\n\x17TYPE_SOLID_SINGLE_WHITE\x10\x02\x12\x1b\n\x17TYPE_SOLID_DOUBLE_WHITE\x10\x03\x12\x1d\n\x19TYPE_BROKEN_SINGLE_YELLOW\x10\x04\x12\x1d\n\x19TYPE_BROKEN_DOUBLE_YELLOW\x10\x05\x12\x1c\n\x18TYPE_SOLID_SINGLE_YELLOW\x10\x06\x12\x1c\n\x18TYPE_SOLID_DOUBLE_YELLOW\x10\x07\x12\x1e\n\x1aTYPE_PASSING_DOUBLE_YELLOW\x10\x08\"H\n\x08StopSign\x12\x0c\n\x04lane\x18\x01 \x03(\x03\x12.\n\x08position\x18\x02 \x01(\x0b\x32\x1c.waymo.open_dataset.MapPoint\":\n\tCrosswalk\x12-\n\x07polygon\x18\x01 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint\":\n\tSpeedBump\x12-\n\x07polygon\x18\x01 \x03(\x0b\x32\x1c.waymo.open_dataset.MapPoint')
)



_TRAFFICSIGNALLANESTATE_STATE = _descriptor.EnumDescriptor(
  name='State',
  full_name='waymo.open_dataset.TrafficSignalLaneState.State',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_ARROW_STOP', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_ARROW_CAUTION', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_ARROW_GO', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_STOP', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_CAUTION', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_GO', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_FLASHING_STOP', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LANE_STATE_FLASHING_CAUTION', index=8, number=8,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=443,
  serialized_end=683,
)
_sym_db.RegisterEnumDescriptor(_TRAFFICSIGNALLANESTATE_STATE)

_LANECENTER_LANETYPE = _descriptor.EnumDescriptor(
  name='LaneType',
  full_name='waymo.open_dataset.LaneCenter.LaneType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TYPE_UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_FREEWAY', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_SURFACE_STREET', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BIKE_LANE', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1245,
  serialized_end=1338,
)
_sym_db.RegisterEnumDescriptor(_LANECENTER_LANETYPE)

_ROADEDGE_ROADEDGETYPE = _descriptor.EnumDescriptor(
  name='RoadEdgeType',
  full_name='waymo.open_dataset.RoadEdge.RoadEdgeType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TYPE_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_ROAD_EDGE_BOUNDARY', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_ROAD_EDGE_MEDIAN', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1458,
  serialized_end=1546,
)
_sym_db.RegisterEnumDescriptor(_ROADEDGE_ROADEDGETYPE)

_ROADLINE_ROADLINETYPE = _descriptor.EnumDescriptor(
  name='RoadLineType',
  full_name='waymo.open_dataset.RoadLine.RoadLineType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TYPE_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BROKEN_SINGLE_WHITE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_SOLID_SINGLE_WHITE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_SOLID_DOUBLE_WHITE', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BROKEN_SINGLE_YELLOW', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BROKEN_DOUBLE_YELLOW', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_SOLID_SINGLE_YELLOW', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_SOLID_DOUBLE_YELLOW', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_PASSING_DOUBLE_YELLOW', index=8, number=8,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1667,
  serialized_end=1941,
)
_sym_db.RegisterEnumDescriptor(_ROADLINE_ROADLINETYPE)


_MAP = _descriptor.Descriptor(
  name='Map',
  full_name='waymo.open_dataset.Map',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='map_features', full_name='waymo.open_dataset.Map.map_features', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dynamic_states', full_name='waymo.open_dataset.Map.dynamic_states', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=176,
)


_DYNAMICSTATE = _descriptor.Descriptor(
  name='DynamicState',
  full_name='waymo.open_dataset.DynamicState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp_seconds', full_name='waymo.open_dataset.DynamicState.timestamp_seconds', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lane_states', full_name='waymo.open_dataset.DynamicState.lane_states', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=178,
  serialized_end=284,
)


_TRAFFICSIGNALLANESTATE = _descriptor.Descriptor(
  name='TrafficSignalLaneState',
  full_name='waymo.open_dataset.TrafficSignalLaneState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lane', full_name='waymo.open_dataset.TrafficSignalLaneState.lane', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state', full_name='waymo.open_dataset.TrafficSignalLaneState.state', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stop_point', full_name='waymo.open_dataset.TrafficSignalLaneState.stop_point', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _TRAFFICSIGNALLANESTATE_STATE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=287,
  serialized_end=683,
)


_MAPFEATURE = _descriptor.Descriptor(
  name='MapFeature',
  full_name='waymo.open_dataset.MapFeature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='waymo.open_dataset.MapFeature.id', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lane', full_name='waymo.open_dataset.MapFeature.lane', index=1,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='road_line', full_name='waymo.open_dataset.MapFeature.road_line', index=2,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='road_edge', full_name='waymo.open_dataset.MapFeature.road_edge', index=3,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stop_sign', full_name='waymo.open_dataset.MapFeature.stop_sign', index=4,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crosswalk', full_name='waymo.open_dataset.MapFeature.crosswalk', index=5,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='speed_bump', full_name='waymo.open_dataset.MapFeature.speed_bump', index=6,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='feature_data', full_name='waymo.open_dataset.MapFeature.feature_data',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=686,
  serialized_end=1032,
)


_MAPPOINT = _descriptor.Descriptor(
  name='MapPoint',
  full_name='waymo.open_dataset.MapPoint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='waymo.open_dataset.MapPoint.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='waymo.open_dataset.MapPoint.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='waymo.open_dataset.MapPoint.z', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1034,
  serialized_end=1077,
)


_LANECENTER = _descriptor.Descriptor(
  name='LaneCenter',
  full_name='waymo.open_dataset.LaneCenter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='speed_limit_mph', full_name='waymo.open_dataset.LaneCenter.speed_limit_mph', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='waymo.open_dataset.LaneCenter.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='interpolating', full_name='waymo.open_dataset.LaneCenter.interpolating', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='polyline', full_name='waymo.open_dataset.LaneCenter.polyline', index=3,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LANECENTER_LANETYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1080,
  serialized_end=1338,
)


_ROADEDGE = _descriptor.Descriptor(
  name='RoadEdge',
  full_name='waymo.open_dataset.RoadEdge',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='waymo.open_dataset.RoadEdge.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='polyline', full_name='waymo.open_dataset.RoadEdge.polyline', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ROADEDGE_ROADEDGETYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1341,
  serialized_end=1546,
)


_ROADLINE = _descriptor.Descriptor(
  name='RoadLine',
  full_name='waymo.open_dataset.RoadLine',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='waymo.open_dataset.RoadLine.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='polyline', full_name='waymo.open_dataset.RoadLine.polyline', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ROADLINE_ROADLINETYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1549,
  serialized_end=1941,
)


_STOPSIGN = _descriptor.Descriptor(
  name='StopSign',
  full_name='waymo.open_dataset.StopSign',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lane', full_name='waymo.open_dataset.StopSign.lane', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='position', full_name='waymo.open_dataset.StopSign.position', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1943,
  serialized_end=2015,
)


_CROSSWALK = _descriptor.Descriptor(
  name='Crosswalk',
  full_name='waymo.open_dataset.Crosswalk',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='polygon', full_name='waymo.open_dataset.Crosswalk.polygon', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2017,
  serialized_end=2075,
)


_SPEEDBUMP = _descriptor.Descriptor(
  name='SpeedBump',
  full_name='waymo.open_dataset.SpeedBump',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='polygon', full_name='waymo.open_dataset.SpeedBump.polygon', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2077,
  serialized_end=2135,
)

_MAP.fields_by_name['map_features'].message_type = _MAPFEATURE
_MAP.fields_by_name['dynamic_states'].message_type = _DYNAMICSTATE
_DYNAMICSTATE.fields_by_name['lane_states'].message_type = _TRAFFICSIGNALLANESTATE
_TRAFFICSIGNALLANESTATE.fields_by_name['state'].enum_type = _TRAFFICSIGNALLANESTATE_STATE
_TRAFFICSIGNALLANESTATE.fields_by_name['stop_point'].message_type = _MAPPOINT
_TRAFFICSIGNALLANESTATE_STATE.containing_type = _TRAFFICSIGNALLANESTATE
_MAPFEATURE.fields_by_name['lane'].message_type = _LANECENTER
_MAPFEATURE.fields_by_name['road_line'].message_type = _ROADLINE
_MAPFEATURE.fields_by_name['road_edge'].message_type = _ROADEDGE
_MAPFEATURE.fields_by_name['stop_sign'].message_type = _STOPSIGN
_MAPFEATURE.fields_by_name['crosswalk'].message_type = _CROSSWALK
_MAPFEATURE.fields_by_name['speed_bump'].message_type = _SPEEDBUMP
_MAPFEATURE.oneofs_by_name['feature_data'].fields.append(
  _MAPFEATURE.fields_by_name['lane'])
_MAPFEATURE.fields_by_name['lane'].containing_oneof = _MAPFEATURE.oneofs_by_name['feature_data']
_MAPFEATURE.oneofs_by_name['feature_data'].fields.append(
  _MAPFEATURE.fields_by_name['road_line'])
_MAPFEATURE.fields_by_name['road_line'].containing_oneof = _MAPFEATURE.oneofs_by_name['feature_data']
_MAPFEATURE.oneofs_by_name['feature_data'].fields.append(
  _MAPFEATURE.fields_by_name['road_edge'])
_MAPFEATURE.fields_by_name['road_edge'].containing_oneof = _MAPFEATURE.oneofs_by_name['feature_data']
_MAPFEATURE.oneofs_by_name['feature_data'].fields.append(
  _MAPFEATURE.fields_by_name['stop_sign'])
_MAPFEATURE.fields_by_name['stop_sign'].containing_oneof = _MAPFEATURE.oneofs_by_name['feature_data']
_MAPFEATURE.oneofs_by_name['feature_data'].fields.append(
  _MAPFEATURE.fields_by_name['crosswalk'])
_MAPFEATURE.fields_by_name['crosswalk'].containing_oneof = _MAPFEATURE.oneofs_by_name['feature_data']
_MAPFEATURE.oneofs_by_name['feature_data'].fields.append(
  _MAPFEATURE.fields_by_name['speed_bump'])
_MAPFEATURE.fields_by_name['speed_bump'].containing_oneof = _MAPFEATURE.oneofs_by_name['feature_data']
_LANECENTER.fields_by_name['type'].enum_type = _LANECENTER_LANETYPE
_LANECENTER.fields_by_name['polyline'].message_type = _MAPPOINT
_LANECENTER_LANETYPE.containing_type = _LANECENTER
_ROADEDGE.fields_by_name['type'].enum_type = _ROADEDGE_ROADEDGETYPE
_ROADEDGE.fields_by_name['polyline'].message_type = _MAPPOINT
_ROADEDGE_ROADEDGETYPE.containing_type = _ROADEDGE
_ROADLINE.fields_by_name['type'].enum_type = _ROADLINE_ROADLINETYPE
_ROADLINE.fields_by_name['polyline'].message_type = _MAPPOINT
_ROADLINE_ROADLINETYPE.containing_type = _ROADLINE
_STOPSIGN.fields_by_name['position'].message_type = _MAPPOINT
_CROSSWALK.fields_by_name['polygon'].message_type = _MAPPOINT
_SPEEDBUMP.fields_by_name['polygon'].message_type = _MAPPOINT
DESCRIPTOR.message_types_by_name['Map'] = _MAP
DESCRIPTOR.message_types_by_name['DynamicState'] = _DYNAMICSTATE
DESCRIPTOR.message_types_by_name['TrafficSignalLaneState'] = _TRAFFICSIGNALLANESTATE
DESCRIPTOR.message_types_by_name['MapFeature'] = _MAPFEATURE
DESCRIPTOR.message_types_by_name['MapPoint'] = _MAPPOINT
DESCRIPTOR.message_types_by_name['LaneCenter'] = _LANECENTER
DESCRIPTOR.message_types_by_name['RoadEdge'] = _ROADEDGE
DESCRIPTOR.message_types_by_name['RoadLine'] = _ROADLINE
DESCRIPTOR.message_types_by_name['StopSign'] = _STOPSIGN
DESCRIPTOR.message_types_by_name['Crosswalk'] = _CROSSWALK
DESCRIPTOR.message_types_by_name['SpeedBump'] = _SPEEDBUMP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Map = _reflection.GeneratedProtocolMessageType('Map', (_message.Message,), {
  'DESCRIPTOR' : _MAP,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Map)
  })
_sym_db.RegisterMessage(Map)

DynamicState = _reflection.GeneratedProtocolMessageType('DynamicState', (_message.Message,), {
  'DESCRIPTOR' : _DYNAMICSTATE,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.DynamicState)
  })
_sym_db.RegisterMessage(DynamicState)

TrafficSignalLaneState = _reflection.GeneratedProtocolMessageType('TrafficSignalLaneState', (_message.Message,), {
  'DESCRIPTOR' : _TRAFFICSIGNALLANESTATE,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.TrafficSignalLaneState)
  })
_sym_db.RegisterMessage(TrafficSignalLaneState)

MapFeature = _reflection.GeneratedProtocolMessageType('MapFeature', (_message.Message,), {
  'DESCRIPTOR' : _MAPFEATURE,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.MapFeature)
  })
_sym_db.RegisterMessage(MapFeature)

MapPoint = _reflection.GeneratedProtocolMessageType('MapPoint', (_message.Message,), {
  'DESCRIPTOR' : _MAPPOINT,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.MapPoint)
  })
_sym_db.RegisterMessage(MapPoint)

LaneCenter = _reflection.GeneratedProtocolMessageType('LaneCenter', (_message.Message,), {
  'DESCRIPTOR' : _LANECENTER,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.LaneCenter)
  })
_sym_db.RegisterMessage(LaneCenter)

RoadEdge = _reflection.GeneratedProtocolMessageType('RoadEdge', (_message.Message,), {
  'DESCRIPTOR' : _ROADEDGE,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.RoadEdge)
  })
_sym_db.RegisterMessage(RoadEdge)

RoadLine = _reflection.GeneratedProtocolMessageType('RoadLine', (_message.Message,), {
  'DESCRIPTOR' : _ROADLINE,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.RoadLine)
  })
_sym_db.RegisterMessage(RoadLine)

StopSign = _reflection.GeneratedProtocolMessageType('StopSign', (_message.Message,), {
  'DESCRIPTOR' : _STOPSIGN,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.StopSign)
  })
_sym_db.RegisterMessage(StopSign)

Crosswalk = _reflection.GeneratedProtocolMessageType('Crosswalk', (_message.Message,), {
  'DESCRIPTOR' : _CROSSWALK,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Crosswalk)
  })
_sym_db.RegisterMessage(Crosswalk)

SpeedBump = _reflection.GeneratedProtocolMessageType('SpeedBump', (_message.Message,), {
  'DESCRIPTOR' : _SPEEDBUMP,
  '__module__' : 'waymo_open_dataset.protos.map_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.SpeedBump)
  })
_sym_db.RegisterMessage(SpeedBump)


# @@protoc_insertion_point(module_scope)
