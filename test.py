import lib_3dai
import config

lib_3dai.generate_patterns(config.PROJECTOR_RES,config.PATTERN_DIR,config.PHASES_DEG,config.NUM_FRINGES)

lib_3dai.get_patterns(config.PATTERN_DIR)

lib_3dai.capture_kinect("test",config.CAPTURE_DIR,config.ROI_Y,config.ROI_X)
