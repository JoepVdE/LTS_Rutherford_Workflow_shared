*SET_PART_LIST
        nn $(should be 1 higher than the last segment generated)
        $here, list all parts, except the two last planes
        24        19        18        17        16        15        14        13
        12        11        10         9         8         7         6         5
         4         3         2         1        21        20        25
*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID
$       ID                                                               heading
       140                                                      Body Interaction
$     ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr
        nn         0         2         2         0         0         0         0
$       fs        fd        dc        vc       vdc    penchk        bt        dt
         0         0         0         0        10         0         0         0
$      sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf
         0         0         0         0         0         0         0         0
$     soft   softscl    lcidab    maxpar     sbopt     depth     bsort    frcfrq
         2         0         0         0         3         5         0         0
$   penmax    tkhopt    shlthk     snlog      isym     i2d3d    sldthk    sldstf
         0         0         0         0         0         0         0         0
*SET_PART_LIST
        nn $nn = nn+1
        $here, list all parts, except the first two planes.
        23         9        10        12        11        13        14        15
        16        17        18        19        20        21         1         3
         2         4         5         6         7         8        22
*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID
$       ID                                                               heading
       150                                                    Body Interaction 2
$     ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr
        nn         0         2         2         0         0         0         0
$       fs        fd        dc        vc       vdc    penchk        bt        dt
         0         0         0         0        10         0         0         0
$      sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf
         0         0         0         0         0         0         0         0
$     soft   softscl    lcidab    maxpar     sbopt     depth     bsort    frcfrq
         2         0         0         0         3         5         0         0
$   penmax    tkhopt    shlthk     snlog      isym     i2d3d    sldthk    sldstf
         0         0         0         0         0         0         0         0