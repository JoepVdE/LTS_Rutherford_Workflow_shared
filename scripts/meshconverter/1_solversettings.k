$# LS-DYNA Keyword file created by Joeps script
*KEYWORD
*PARAMETER_DUPLICATION
4
*DATABASE_FORMAT
$    iform   ibinary                                                      unused
         0         0                                                            
*CONTROL_TERMINATION
$   endtim    endcyc     dtmin    endeng    endmas                        unused
     0.001  10000000     0.001         0    100000                              
*CONTROL_TIMESTEP
$   dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st
         0       0.9         0         0         0         0         1         0
$   dt2msf   dt2mslc     imscl                                            unused
         0         0         0                                                  
*CONTROL_CONTACT
$   slsfac    rwnpal    islchk    shlthk    penopt    thkchg     orien    enmass
         0         0         1         1         1         0         2         0
$   usrstr   usrfric     nsbcs    interm     xpene     ssthk      ecdt   tiedprj
         0         0         0         0         0         0         0         0
$    sfric     dfric       edc       vfc        th     th_sf    pen_sf    unused
         0         0         0         0         0         0         0          
$   ignore    frceng   skiprwg    outseg   spotstp   spotdel   spothin   unused1
         2         1         0         1         0         1       0.5          
$     isym    nserod    rwgaps    rwgdth     rwksf      icov    swradf    ithoff
         0         0         0         0         0         0         0         0
$   shledg    pstiff    ithcnt    tdcnof     ftall                       unused2
         1         0         0         0         1                              
*CONTROL_ENERGY
$     hgen      rwen    slnten     rylen                                  unused
         2         2         2         2                                        
*CONTROL_ACCURACY
$      osu       inn    pidosu                                            unused
         0         1         0                                                 0
*CONTROL_BULK_VISCOSITY
$       q1        q2      type                                            unused
       1.5      0.06        -2                                                  
*CONTROL_SOLID
$    esort    fmatrx   niptets    swlocl                                  unused
         1         0         0         0                                        
$    pm1     pm2     pm3     pm4     pm5     pm6     pm7     pm8     pm9    pm10
       0       0       0       0       0       0       0       0       0       0
*CONTROL_SHELL
$   wrpang     esort     irnxx    istupd    theory       bwc     miter      proj
        20         1        -1         4         2         1         1         1
$  rotascl    intgrd    lamsht    cstyp6    tshell    nfail1    nfail4   psnfail
         0         0         0         0         0         1         1         0
$   pstupd    irquad     cntco                                            unused
         0         0         2                                                  
*CONTROL_OUTPUT
$    npopt    neecho    nrefup    iaccop     opifs    ipnint    ikedit    iflush
         1         0         0         0         0         0         0         0
$    iprtf    ierode     tet10    msgmax    ipcurv                        unused
         0         0         0         0         0                              
*CONTROL_PARALLEL
$     ncpu    numrhs    consti      para                                  unused
         0         0         1         0                                        
*CONTROL_SOLUTION
$     soln       nlq     isnan     lcint                                  unused
         0         0         0         0                                        
*CONTROL_HOURGLASS
$      ihq        qh                                                     unused1
         5       0.1          