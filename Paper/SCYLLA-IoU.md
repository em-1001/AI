### **SCYLLA-IoU(SIoU)**

SCYLLA-IoU (SIoU) considers **Angle cost**, **Distance cost**, **Shape cost** and the penalty term is as follows.

$$
\mathcal{R}_{SIoU} = \frac{\Delta + \Omega}{2}
$$

**Angle cost**

Angle cost is calculated as follows

$$
\begin{aligned}\Lambda &= 1 - 2 \cdot \sin^2\left(\arcsin(x) - \frac{\pi}{4} \right) \\   &= 1 - 2 \cdot \sin^2\left(\arcsin(\sin(\alpha)) - \frac{\pi}{4} \right) \\&= 1 - 2 \cdot \sin^2\left(\alpha - \frac{\pi}{4} \right) \\&= \cos^2\left(\alpha - \frac{\pi}{4}\right) - \sin^2\left(\alpha - \frac{\pi}{4}\right) \\ &= \cos\left(2\alpha - \frac{\pi}{2}\right) \\ &= \sin(2\alpha) \\ \end{aligned}
$$

$$
\begin{aligned} &where \\   &x = \frac{c_h}{\sigma} = \sin(\alpha) \\  &\sigma = \sqrt{(b_{c_x}^{gt} - b_{c_x})^2 + (b_{c_y}^{gt} - b_{c_y})^2} \\  &c_h = \max(b_{c_y}^{gt}, b_{c_y}) - \min(b_{c_y}^{gt}, b_{c_y})\end{aligned}
$$

If  $\alpha > \frac{\pi}{4}$ , then $\beta = \frac{\pi}{2} - \alpha$, which is calculated as beta.

**Distance cost**

Distance cost includes Angle cost, which is calculated as follows

$$
\begin{aligned}&\Delta = \sum_{t=x,y} (1 - e^{-\gamma \rho_t}) \\  &where \\  &\rho_ x = \left(\frac{b_{c_x}^{gt} - b_{c_x}}{c_w} \right)^2, \ \rho_ y = \left(\frac{b_{c_y}^{gt} - b_{c_y}}{c_h} \right)^2, \ \gamma = 2 - \Lambda\end{aligned}
$$

Here, $c_w, c_h$ are the width and height of the smallest box containing $B$ and $B^{gt}$, unlike the Angle cost.

If we look at the Distance cost, we can see that it gets sharply smaller as $\alpha \to 0$ and larger as $\alpha \to \frac{\pi}{4}$, so $\gamma$ is there to adjust it.

**Shape cost**

Shape cost is calculated as follows

$$
\begin{aligned}&\Omega = \sum_{t=w,h} (1-e^{-\omega_t})^{\theta} \\ &\\ &where \\ &\\  &\omega_w = \frac{|w-w^{gt}|}{\max(w,w^{gt})}, \omega_h = \frac{|h-h^{gt}|}{\max(h,h^{gt})} \\   \end{aligned}
$$

The $\theta$ specifies how much weight to give to the Shape cost, usually set to 4 and can be a value between 2 and 6.

The final loss is

$$
L_{SIoU} = 1 - IoU + \frac{\Delta + \Omega}{2}
$$
