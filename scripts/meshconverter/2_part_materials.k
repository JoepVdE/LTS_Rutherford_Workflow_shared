$for each wire n and plate (n+m)



$elastic materials n
$start here
*PART
$                                                             partname   unused1
wire_n
$       ID     secid       mid     eosid      hgid      grav     adopt      tmid
         n         n         n         0         0         0         0         0
*SECTION_SOLID
$       ID    elform       aet                                           unused1
         n         1         0                                                                                                    
*MAT_PLASTIC_KINEMATIC
$       ID        ro         E        pr      sigy      etan      beta   unused1
         n  7.85E-09    200000       0.3      sigy      etan         0          
$      src       srp        fs        vp                                 unused2
         0         0         0         0                                        
$Material Id = {n,{150AACD8-78AF-4983-8776-AA8DC3BA91BD}, Copper Wire}
$Wire material: sigy=sigy_comment MPa, etan=etan_comment MPa (from cable_parameters_user.json wire_material)
$end here



$rigid materials m
$start here
*PART
$                                                             partname   unused1
plate_m
$       ID     secid       mid     eosid      hgid      grav     adopt      tmid
        m        m        m         0         0         0         0         0
*SECTION_SOLID
$       ID    elform       aet                                           unused1
         m        1          0                                                                                                    
*MAT_RIGID
$       ID        ro         E        pr         n    couple         m     alias
        m  7.85E-09    200000       0.3         0         0         0         0
$      cmo      con1      con2                                           unused1
         1         5         7                                                  
$      lco        A2        A3        V1        V2        V3             unused2
         0         0         0         0         0         0                    
$Material Id = {25,e37a0f0f-25eb-46e6-ba05-9e2d08eafc9e, Structural Steel}
$Material Unique Id = {25,59002a79-71c2-428e-906b-55d7dbee100e, Structural Steel}
$Material Id = {25,{275D7C29-3F25-4B36-BC5D-F5067FA02501}, Structural Steel}
$end here

