struct AdamOptimizer {
}

struct CheckConfig {
    default: bool,
    alpha:f64,
    beta1:f64,
    beta2:f64,
    epsilon:f64,
    epoch:f64
}

struct GradientDescent {
}

//impl CheckConfig {
//    fn is_default(&self) {
//        if self.default == true {
//            let (alpha, beta1, beta2, epsilon, epoch) = (0.001, 0.9, 0.999, 0.1, 10);
//            println!("Using default configs. alpha: {}, beta1: {}, beta2: {}, epsilon: {}", alpha, beta1, beta2, epsilon, epoch);
//        } else {
//            let (alpha, beta1, beta2, epsilon, epoch) = (self.alpha, self.beta1, self.beta2, self.epsilon);
//            println!("Using custom configs. alpha: {}, beta1: {}, beta2: {}, epsilon: {}", alpha, beta1, beta2, epsilon, epoch);
//        }
//    }
//}

impl AdamOptimizer {
    fn is_default(&self, default: bool, alpha: f64, beta1: f64, beta2: f64, epsilon: f64, epoch: i64) {
        if default == true {
            let (alpha, beta1, beta2, epsilon, epoch) = (0.001, 0.9, 0.999, 0.1, 10);
            println!("Using default configs. alpha: {}, beta1: {}, beta2: {}, epsilon: {}, epoch: {}", alpha, beta1, beta2, epsilon, epoch,)
            //backwards_propagation(alpha, beta1, beta2, epsilon,)
        } else {
            let (alpha, beta1, beta2, epsilon, epoch) = (alpha, beta1, beta2, epsilon, epoch);
            println!("Using custom configs. alpha: {}, beta1: {}, beta2: {}, epsilon: {}, epoch: {}", alpha, beta1, beta2, epsilon, epoch)
            //backwards_propagation(alpha, beta1, beta2, epsilon, epoch, gradient, weights)
        }
        //Make sure to add all the necessary variables for functions below
    }
    fn backwards_propagation(&self, alpha: f64, beta1: f64, beta2: f64, epsilon: f64, epoch: i64, gradient: Vec<f64>, theta: Vec<f64>) -> std::vec::Vec<f64> {
        //println!("Loading configs. alpha: {}, beta1: {}, beta2: {}, epsilon: {}, epoch: {}", alpha, beta1, beta2, epsilon, epoch);
        // x = input 1 vector
        // y = input 2 vector
        let m = 0.0;
        let v = 0.0;
        let t: i32 = 0;
        let theta = theta;
        let t_bp: i32 = t+1;
        let m_bp = vec![beta1*m + (1.0-beta1)*gradient[0], beta1*m + (1.0-beta1)*gradient[1]];
        let v_bp = vec![beta2*v+(1.0-beta2)*(gradient[0].powi(2)), beta2*v+(1.0-beta2)*(gradient[1].powi(2))];
        let m_hat = vec![m_bp[0]/(1.0- beta1.powi(t_bp)), m_bp[1]/(1.0- beta1.powi(t_bp))];
        let v_hat = vec![v_bp[0]/(1.0- beta2.powi(t_bp)), v_bp[1]/(1.0- beta2.powi(t_bp))];
        let mut theta_bp = vec![0.0, 0.0];
        theta_bp[0] += theta[0] - alpha*(m_hat[0]/v_hat[0].sqrt() - epsilon);
        theta_bp[1] += theta[1] - alpha*(m_hat[0]/v_hat[1].sqrt() - epsilon);
        theta_bp as Vec<f64>
    }
}

impl GradientDescent {
    fn cost_function(&self, x: &Vec<f64>, y: &Vec<f64>, theta_0: f64, theta_1: f64) -> f64 {
        let m = x.len() as f64;
        let mut v: f64 = 0.0;
        for i in x {
            let h = theta_0 + theta_1 * i;
            v += (h - i).powi(2);
        }
        v/(2.0*m)
    }
    fn gradient_function(&self, x: &f64, y: &f64, theta_0: &f64, theta_1: &f64) -> Vec<f64> {
        let mut v = vec![0.0, 0.0];
        let h = theta_0 + theta_1 * x;
        v[0] += h - y;
        v[1] += (h - y)*x;
        v
    }
}

fn main() {
    let x = vec![1.0, 2.1, 3.9, 4.2, 5.1];
    let y = vec![1.0, 2.1, 3.9, 4.2, 5.1];
    let print_interval = 100;
    let initial_theta = vec![0.0, 0.0];
    let m = x.len().clone();

    let default = true;
    let (alpha, beta1, beta2, epsilon, epoch) = (0.001, 0.9, 0.999, 0.1, 10);

    //let batch_size = // some vector
    let theta = vec![0.0, 0.0];
    //let lambda_h = |theta_0, theta_1, x| theta_0+theta_1*x; Old closure
    //let weights = theta;
    let runner = AdamOptimizer{};
    runner.is_default(default, alpha, beta1, beta2, epsilon, epoch);
    let gd = GradientDescent{};
    let initial_cost = gd.cost_function(&x.clone(), &y.clone(), initial_theta[0], initial_theta[1]);
    let initial_cost_vector = vec![initial_cost];
    let mut history = Vec::new();
    history.extend([theta.clone(), initial_cost_vector].iter().cloned());

    for j in 0..epoch {
        for i in 0..m {
            let gradient = gd.gradient_function(&x[i], &y[i], &initial_theta[0], &initial_theta[1]);
            let theta = runner.backwards_propagation(alpha, beta1, beta2, epsilon, epoch, gradient, theta.clone());

        if (j+1)%print_interval == 0 {
            let cost = gd.cost_function(&x, &y, initial_theta[0], initial_theta[1]);
            let cost_vector = vec![cost];
            history.extend([theta, cost_vector].iter().cloned());
            println!("{:?}", history)

        } else if j==0 {
            let cost = gd.cost_function(&x, &y, initial_theta[0], initial_theta[1]);
            let cost_vector = vec![cost];
            history.extend([theta, cost_vector].iter().cloned());
            println!("{:?}", history)
            }
        }
    }

}
