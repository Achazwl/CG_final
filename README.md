# Intersection

## Sphere

限制： $\|d\|^2=1$

$$
\begin{aligned}
\|o + d * t - c\|^2 &= r ^ 2\\
[\|d\|^2] t^2 + [2 d \cdot (o-c)] t + [\|o-c\|^2-r^2] &= 0\\
t^2 + 2 b t + c &= 0\\
\end{aligned}
$$

于是

$$
\begin{aligned}
t &= {2b \pm \sqrt{(2b)^2 - 4 * 1 * c}\over 2*1} \\
&= b \pm \sqrt {b^2-c}
\end{aligned}
$$

## Acceleration

### Space partition (KD-tree)

常规KD-tree，叶子节点暴力逐个求交

利用空间关系和光线方向，如果要递归两边则先递归靠近光源的一侧，这要一交到就不用另一侧递归了直接回溯了

### Object partition (BVH)

bounding volume hierarchy

父包围盒子完全包含子包围盒， 但子包围盒可以相交

每个obj属于且仅属于一个叶子节点

永远选长轴切半，尽量对半分

box里物体很少时，不再递归，暴力

![image-20210525102053196](/home/acha/cv_big/README.assets/image-20210525102053196.png)

#### SAH

基于表面积的启发式评估划分方法（Surface Area Heuristic，SAH）

​	

### compare

> ​	First, it is important to observe that there are basically two different "classes" of methods for building accelerations structures for ray tracing: 1) space subdivision methods (e.g. BSP, kD-Tree, Octree, etc.); and 2) object subdivision methods (e.g. BVH). While space subdivison methods work by recursively subdividing the space with planes, without sticking to the geometry embedded into that space, object subdivison methods work by subdividing the geometry recursively into smaller and smaller parts, wrapping each part with a, usually tight, volume. More on the basics of space and object subdivision that can be found in the PBRT v3 online book.
​	Each method (space or object sudivision) has its own pros and cons. In the case of space subdivision methods, since the subspaces do not overlap, we can usually traverse the structure in front-to-back, or back-to-front, order more easily. When a ray is traversing such structure, as soon as it hits a surface, we can stop the traversal. This usually leads to faster traversal schemes. Several software renderers take advantage of the traversal efficiency given by the space subdivision schemes. On the other hand, space subdivision schemes may be more intricate to implement (usually you have to tweak some epsilons) and may lead to deeper trees. Also, they do not like very much dynamic geometry. If the geometry encoded into a space subdivision acceleration structure changes, we usually have to rebuild the acceleration structure from scratch.
​	Object subdivision methods have quite different characteristics. Since the object is subdivided with volumes, and these volumes may overlap, the traversal is traditionally slower. We cannot, for instance, stop traversing a BVH as soon as a ray finds an intersection with a surface. Since the volumes overlap, we may need to check for potential intersections with nearby primitives before quitting the traversal. On the other hand, it may be easier to implement a BVH because we do not have to split the object parts with planes. Also, BVHs usually generate shallower structures (which may eventually compensate the slower traversal). However, one of the most interesting aspects of the BVHs is that they are dynamic geometry-friendly. If geometry changes (but not much, actually), we can simply locally adjust the size and position of the corresponding bounding volume (i.e. refitting). These adjustments may cause the need to adjust the parent volumes, a procedure that may culminate in a chain reaction that may reach the root node of the BVH. All in all, assuming that we have a reasonably balanced BVH, these operations will be ~O(log n), which is really fast and cool.

## Octree

# Ray

递归一定层数之后， 开启俄罗斯轮盘赌, 使得有限步就会收敛到结束
以最大的颜色值为p，大于p直接递归，否则f/=p, 期望仍为 p*(f/p)=f

## DIFFUSE

随机单位半球面上一点，往该方向反射
先随机底面2pi往哪个方向
再随机底面半径(这样得到的并不是那么均匀, 但是更真实)
TODO: From Realistic Ray Tracing (Shirley and Morley)

随便乱射有时很难射到光源上， 改为：直接遍历查找能直接照到的，能照到的直接照，需要间接的才递归，这样对于小光源有很大的效率提升
首先计算直射光源球体中心, 以及射线切线球体时 中间的夹角 $\alpha_{max}$
然后再随机更小的角度（这里也不是均匀的，但是更真实）
按照极座标生成射线
注意，这里的计算由于一次性综合考虑了多个光源，需要模拟积分
顶角为$2\theta$ 的圆锥的立体角为 $2\pi (1-\cos \theta)$

## MIRROR

入射角=反射角

记 $n_l$ 为光线照射面的法向（与light的方向相反）
$-d\cdot n_l$ 即为 $\cos i$

$d + 2 n_l \cos_i$ 即得 反射光方向

## GLASS

首先判断光线是从内部往外射还是外部往内射
从而得出入射界面的折射率 $n_a$ 和 另一侧的折射率 $n_b$

折射公式：

若 ${n_a\over n_b}\sin i > 1$ 则超过临界角，发生全反射，按照mirror的方式去做

否则 $\sin r = {n_a\over n_b}\sin i$

反射光在界面的方向(normalize后)为 ${d+n_l\cos i\over \sin i}$

因此折射光的方向为 反射光在界面的方向乘 $\sin r$ 加 反射光在法线方向乘 $\cos r$

$$
\begin{aligned}
rd &= {d+n_l\cos i\over \sin i} \sin r + (-n_l) \cos r\\
&= {n_a\over n_b} (d+n_l\cos i) + (-n_l) \cos r
\end{aligned}
$$


然后要计算反射和折射的强度比
Schlick 菲涅尔近似等式
$反射占比 = F_0 + (1 - F_0) * (1-\cos \theta_i)^5$
其中 $F_0 = {(n_a/n_b-1)^2 \over (n_a/n_b+1)^2}$ (这里a,b交换不影响计算结果)

# Image

## Sample

2*2 subpixel

1. center (of subpixel)
$(0+0.5)/2=0.25$
$(1+0.5)/2=0.75$

2. offset
用filter做sample

3. ray
$(center+offset) / width - 0.5 \in (-0.5, 0.5)$