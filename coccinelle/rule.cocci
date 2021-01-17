@rule@
identifier F1;
type T;
position p1;
@@
*T F1@p1(...)
*{...}

@script:python@
r2f1 << rule.F1;
p1 << rule.p1;
@@
f=open("func_list.txt","a+")
f.write("%s :: %s \n" %(r2f1,p1[0].file))
f.close()
print ("======  %s \n " %(p1[0].file))

