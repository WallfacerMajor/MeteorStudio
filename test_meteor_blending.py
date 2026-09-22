import unittest
from dataclasses import replace

import cv2
import numpy as np

from meteor_blending import blend_signal, remove_mask_stars
from meteor_composer import Stroke, compose_meteor_objects, compose_meteor_sources, transformed_object_crop, stroke_for_image_crop


class BlendingTests(unittest.TestCase):
    def test_formula_depth_alpha_and_clipping(self):
        for maximum in (255, 65535):
            d = np.full((2, 2, 3), maximum*.4)
            s = np.full_like(d, maximum*.8)
            a = np.full((2, 2), .5)
            np.testing.assert_allclose(blend_signal(d,s,a,maximum,'滤色'), maximum*.64, rtol=1e-6)
            np.testing.assert_allclose(blend_signal(d,s,a,maximum,'线性减淡（添加）'), maximum*.7, rtol=1e-6)
            np.testing.assert_allclose(blend_signal(d,s,a*0,maximum,'滤色'),d,rtol=1e-6)

    def test_overlap_and_source_groups(self):
        for dtype, maximum in ((np.uint8,255),(np.uint16,65535)):
            base = np.full((100,150,3),int(maximum*.2), dtype)
            src = base.copy()
            cv2.line(src,(30,50),(120,50),(int(maximum*.5),)*3,3)
            stroke = Stroke([(.2,.505),(.8,.505)], 15, 0, preserve_brightness_override=False)
            for mode in ('滤色','线性减淡（添加）'):
                settings = (False,False,0,0,mode,False,100,0,False)
                one,_ = compose_meteor_objects(src,base,[stroke],*settings)
                both,_ = compose_meteor_objects(src,base,[stroke,replace(stroke)],*settings)
                split,_ = compose_meteor_sources(src,src,base,[stroke,replace(stroke,source_mode='original')],*settings)
                np.testing.assert_array_equal(both,split)
                self.assertGreater(int(both[50,75,0]),int(one[50,75,0]))
                sequential,_ = compose_meteor_objects(src,one,[stroke],*settings,background=base)
                np.testing.assert_array_equal(both,sequential)
                np.testing.assert_array_equal(src[0],base[0])

    def test_star_strength_preserves_core_and_input(self):
        for dtype, maximum in ((np.uint8,255),(np.uint16,65535)):
            src = np.full((100,160,3), maximum//10,dtype)
            cv2.line(src,(20,50),(140,50),(maximum*.7,)*3,3)
            cv2.circle(src,(80,62),2,(maximum,)*3,-1)
            original = src.copy()
            alpha = np.ones((100,160),np.float32)
            cleaned = remove_mask_stars(src,alpha,[(20,50),(140,50)],40,100)
            half = remove_mask_stars(src,alpha,[(20,50),(140,50)],40,50)
            np.testing.assert_array_equal(src,original)
            np.testing.assert_array_equal(cleaned[49:52,20:141],src[49:52,20:141])
            self.assertLess(cleaned[62,80,0], maximum*.3)
            self.assertGreater(half[62,80,0],cleaned[62,80,0])
            np.testing.assert_array_equal(remove_mask_stars(src,alpha,[(20,50),(140,50)],40,0),src)

    def test_off_axis_fragments_and_halo_are_not_partly_inpainted(self):
        for dtype,maximum in ((np.uint8,255),(np.uint16,65535)):
            source=np.full((100,160,3),maximum//10,dtype)
            # A rough hand mask misses the meteor centre by five pixels. Its
            # separated bright knots and halo are compact, like star detections.
            for x in (35,60,90,125):
                cv2.circle(source,(x,55),5,(int(maximum*.45),)*3,-1)
                cv2.circle(source,(x,55),2,(int(maximum*.85),)*3,-1)
            cv2.circle(source,(75,70),2,(maximum,)*3,-1)
            result=remove_mask_stars(source,np.ones((100,160),np.float32),[(20,50),(140,50)],40,100)
            np.testing.assert_array_equal(result[49:61,20:141],source[49:61,20:141])
            self.assertLess(result[70,75,0],maximum*.3)

    def test_contraction_and_transform(self):
        src = np.full((100,160,3),100,np.uint8)
        stroke = Stroke([(.2,.5),(.8,.5)],40,6)
        broad = transformed_object_crop(src,stroke)
        narrow = transformed_object_crop(src,replace(stroke,mask_choke=60))
        self.assertLess(narrow[1].sum(),broad[1].sum()*.6)
        shifted = transformed_object_crop(src,replace(stroke,mask_choke=60,offset_y=10))
        self.assertGreater(shifted[3][1],narrow[3][1])

    def test_transformed_crop_matches_full_compositor(self):
        src = np.full((300,400,3),30,np.uint8)
        cv2.line(src,(130,120),(230,150),(180,210,170),4)
        cv2.circle(src,(170,147),3,(220,220,220),-1)
        base=np.full_like(src,30)
        for mode in ('滤色','线性减淡（添加）'):
            stroke=Stroke([(130/399,120/299),(230/399,150/299)],25,4,
                rotation=12,offset_x=13,star_removal=80,mask_choke=25)
            settings=(False,False,0,0,mode,True,100,0,False)
            full,_=compose_meteor_objects(src,base,[stroke],*settings)
            cropped=stroke_for_image_crop(stroke,400,300,50,30,300,240)
            local,_=compose_meteor_objects(src[30:270,50:350],base[30:270,50:350],[cropped],*settings)
            self.assertLessEqual(np.abs(full[30:270,50:350].astype(int)-local.astype(int)).max(),1)

if __name__ == '__main__':
    unittest.main()
