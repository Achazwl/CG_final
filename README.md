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