/*
Code for solving a general stochastic differential equation (SDE)
dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t    where W_t is the Weiner process

e.g. an Ornstein-Uhlenbeck process
dX_t = theta*(mu - X_t)*dt + sigma*dW_t

Compile with:
g++ -std=c++11 ito.cpp -o ito

Daniel J. Sharpe
May 2018
*/

#include <random>
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <fstream>
using namespace std;

#define OUTFILE "sde_soln" // output file
#define RANDSEED 17

/* set simulation params */
#define NDIM 3 // dimensionality of X_t
#define NPOINTS 100 // points to simulate
#define DELTAT 0.01 // timestep
double X_INIT[NDIM] = {6.0, 6.0, 6.0}; // initial position
#define T_INIT 0.0; // initial time

/* set Ornstein-Uhlenbeck params */
double theta[NDIM] = {0.7, 0.65, 0.6}; // mean reversion speed
double sigma_const[NDIM] = {0.06, 0.055, 0.05}; // long-term mean
double mu_const[NDIM] = {1.5, 1.4, 1.3}; // volatility

double ran_no(double min, double max);
double *ou_mu(double **mu_method_args, double *X_t, double t);
double *ou_sigma(double **sigma_method_args, double *X_t, double t);
double *euler_maruyama(double *x_i, double *(*mu_method)(double **, double *, double), \
                       double *(*sigma_method)(double **, double *, double), \
                       double **mu_method_args, double **sigma_method_args, double dt, double t);
double *milstein(double *x_i, double *(*mu_method)(double **, double *, double), \
                 double *(*sigma_method)(double **, double *, double), \
                 double **mu_method_args, double **sigma_method_args, double dt, double t);
double *cen_diff(double *x_i, double *(*func)(double **, double *, double), double **func_args, double t, double h=0.001);

/* driver function */
int main(void)
{
    int i, j;
    double t;
    std::string method = "milstein"; // choose method by which to solve the SDE
    double *x_i = X_INIT;
    double *(*sde_solver) (double *, double *(*)(double **, double *, double), \
                           double *(*)(double **, double *, double), double **, double **, double, double);
    double *(*mu_method) (double **, double *, double);
    double *(*sigma_method) (double **, double *, double);

    std :: cout << "Starting simulation...\n";
    fstream outputfile;
    outputfile.open (OUTFILE, ios::out | ios::app);
    /* Example: an Ornstein-Uhlenbeck process with params defined above */
    mu_method = &ou_mu; // function mu(X_t,t)
    sigma_method = &ou_sigma; // function sigma(X_t,t)
    double *mu_method_args[2] = {&theta[0],&mu_const[0]}; // params required by function mu(X_t,t)
    double *sigma_method_args[1] = {&sigma_const[0]}; // params required by function sigma(X_t,t)
    t = T_INIT;
    for (i=1;i<=NPOINTS;i++) {
        t += DELTAT;
        std :: cout << "iteration " << i << "\n";
        if (method == "eulermar") {
            sde_solver = &euler_maruyama;
        }
        else if (method == "milstein") {
            sde_solver = &milstein;
        }
        x_i = (*sde_solver)(x_i,mu_method,sigma_method,mu_method_args,sigma_method_args,DELTAT,t);
        for (j=0;j<NDIM;j++) {
            outputfile << std :: setprecision(5)  << *(x_i+j) << "  ";
        }
        outputfile << "\n";
    }
    outputfile.close();
    return 0;
}

/* Ornstein-Uhlenbeck process value of the mu(X_t,t) function */
double *ou_mu(double **ou_mu_args, double *X_t, double t)
{
    int i;
    static double mu_val[NDIM];
    double *theta, *mu_const;

    theta = *(ou_mu_args);
    mu_const = *(ou_mu_args+1);

    for (i=0;i<NDIM;i++) {
        mu_val[i] = (*(theta+i))*((*(mu_const+i)) - X_t[i]);
    }

    return mu_val;
}

/* Ornstein-Uhlenbeck process value of the sigma(X_t,t) function */
double *ou_sigma(double **ou_sigma_args, double *X_t, double t)
{
    return *ou_sigma_args;
}

/* Euler-Maruyama method */
double *euler_maruyama(double *x_i, double *(*mu_method)(double **, double *, double), \
                       double *(*sigma_method)(double **, double *, double), \
                       double **mu_method_args, double **sigma_method_args, double dt, double t)
{
    int i;
    double dw_t[NDIM];
    double *mu_val, *sigma_val;

    mu_val = (*mu_method)(mu_method_args,x_i,t);
    sigma_val = (*sigma_method)(sigma_method_args,x_i,t);

    for (i=0;i<NDIM;i++) {
        dw_t[i] = ran_no(0,1);
        x_i[i] = (mu_val[i]*dt) + (sigma_val[i]*dw_t[i]);
    }
    return x_i;
}

/* Milstein method */
double *milstein(double *x_i, double *(*mu_method)(double **, double *, double), \
                 double *(*sigma_method)(double **, double *, double), \
                 double **mu_method_args, double **sigma_method_args, double dt, double t)
{
    int i;
    double dw_t[NDIM];
    double *mu_val, *sigma_val, *sigma_deriv_val;

    mu_val = (*mu_method)(mu_method_args,x_i,t);
    sigma_val = (*sigma_method)(sigma_method_args,x_i,t);
    sigma_deriv_val = cen_diff(x_i,sigma_method,sigma_method_args,t);

    for (i=0;i<NDIM;i++) {
        dw_t[i] = ran_no(0,1);
        x_i[i] = (mu_val[i]*dt) + (sigma_val[i]*dw_t[i]) + \
                 ((0.5*sigma_val[i]*sigma_deriv_val[i])*((pow(dw_t[i],2.0))-dt));
    }
    return x_i;
}

/* central finite difference method (first derivatives wrt components of X_t) */
double *cen_diff(double *x_i, double *(*func)(double **, double *, double), \
                 double **func_args, double t, double h)
{
    int i;
    double forward_funcval[NDIM], backward_funcval[NDIM];
    static double cen_diff_deriv[NDIM];

    for (i=0;i<NDIM;i++) {
        double *x_i_forward = x_i;
        double *x_i_backward = x_i;
        x_i_forward[i] += h;
        x_i_backward[i] -= h;
        forward_funcval[i] = *(*func)(func_args,x_i_forward,t);
        backward_funcval[i] = *(*func)(func_args,x_i_backward,t);
        cen_diff_deriv[i] = (forward_funcval[i] - backward_funcval[i]) / (2.0*h);
    }
    return cen_diff_deriv;
}

/* Random number generator (drawing from normal distribution) */
double ran_no(double min, double max)
{
    static std::mt19937 engine(RANDSEED);
    static std::normal_distribution<double> normal;
    using pick = std::normal_distribution<double>::param_type;
    return normal(engine, pick(min, max));
}
