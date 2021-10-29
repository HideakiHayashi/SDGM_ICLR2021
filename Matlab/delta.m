%======================================
%delta
%======================================
function tmp = delta(c,cc,m,mm)
if c == cc && m == mm
    tmp = 1.0;
else
    tmp = 0.0;
end