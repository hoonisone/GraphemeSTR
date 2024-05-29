x=("a" "b" "c")
y=(("a" "b" "c")
 "b1" 
 "c1")
for i in 0 1 2
do
    echo ${x[i]} ${y[i]}
done