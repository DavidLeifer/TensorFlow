#figure out skikit-learn
import sys
import commpy
from bitstring import BitArray

val1 = '10010'
val2 = '01001'

def hammingDistance(val1, val2):
	if len(val1) != len(val2):
		raise ValueError
		return sum(bool(ord(ch1) - ord(ch2)) for ch1, ch2 in zip(s1, s2))

v1=BitArray(bin=val1)
v2=BitArray(bin=val2)

print "Binary form:\t",val1," - ",val2
print "Decimal form:\t",int(val1,2)," - ",int(val2,2)
print "Hamming distance is ",commpy.utilities.hamming_dist(v1,v2)
