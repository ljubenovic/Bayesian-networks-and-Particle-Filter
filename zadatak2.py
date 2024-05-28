#%%
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

#%%
rho = []
theta = []
with open('observations.csv','r') as file:
    reader = csv.reader(file)
    for row in reader:
        rho.append(float(row[0]))
        theta.append(float(row[1]))

N=2000
particles = np.zeros((3,N)) # x,y, pravac kretanja
particles[0,:] = [-2+4*random.random() for i in range(0,len(particles[0,:]))]
particles[1,:] = [-2+4*random.random() for i in range(0,len(particles[1,:]))]
particles[2,:] = [ 2*np.pi*random.random() for i in range(0,len(particles[2,:]))]

v = 0.5
T = 1
#%%
def rho_std(angle_rho,angle_v):
    angle_rho = np.mod(angle_rho,2*np.pi)
    angle_v = np.mod(angle_v,2*np.pi)
    diff = abs(angle_rho-angle_v)
    if (diff>np.pi/2 and diff<=np.pi):
        diff = np.pi-diff
    elif(diff>np.pi and diff<=3*np.pi/2):
        diff = diff-np.pi
    elif(diff>3*np.pi/2 and diff<=2*np.pi):
        diff = 2*np.pi-diff
    return -0.3*2/np.pi*diff+0.6
#%%

mean_rho = np.zeros((1,len(rho)))
mean_theta = np.zeros((1,len(theta)))
rho_std_iter = np.zeros((1,len(rho)))
theta_std_iter = np.zeros((1,len(theta)))

true_x = np.zeros((1,len(rho)))
true_y = np.zeros((1,len(rho)))
est_x = np.zeros((1,len(rho)))
est_y = np.zeros((1,len(rho)))

theta_std = np.pi/36
p_prom = 0.2*np.ones((1,N))
for i in range(len(rho)):
    weights = np.ones(N) / N
    rho_curr = rho[i]
    theta_curr = theta[i]
    for j in range(N):
        tmp = random.random()
        #print(tmp,p_prom[0][j],j)
        if (tmp<p_prom[0][j]):
            p_prom[0,j] = 0.2
            delta = -np.pi/6 + np.pi/3*random.random()
            particles[2,j] = particles[2,j] + delta
            if(particles[2,j]>2*np.pi):
                particles[2,j] = particles[2,j]-2*np.pi
            elif(particles[2,j]<0):
                particles[2,j] = particles[2,j]+2*np.pi
        else:
            p_prom[0,j] = p_prom[0,j] + 0.2
        particles[0,j] = particles[0,j] + v*np.cos(particles[2,j]) + np.random.normal(0,0.3)
        particles[1,j] = particles[1,j] + v*np.sin(particles[2,j]) + np.random.normal(0,0.3)

        theta_est = np.arctan2(particles[1,j],particles[0,j]) + np.random.laplace(scale=theta_std)
        std_rho = rho_std(theta_curr,particles[2,j])
        rho_est = np.sqrt(particles[0,j]**2+particles[1,j]**2)+np.random.normal(scale =std_rho )
        w_rho = np.exp(-(rho_curr-rho_est)**2/(2*std_rho**2))/(2*std_rho*np.pi)
        w_theta = 1/(2*theta_std)*np.exp(-abs(theta_curr-theta_est)/theta_std)
       
        weights[j] = w_rho*w_theta
    
    weights /= np.sum(weights)
    new_particles = np.zeros((3,N))
    new_particles_idx = np.random.choice(N, size=N, replace=True, p=weights)
    new_particles = particles[:, new_particles_idx]
    best_particles_idx = np.argsort(weights)[::-1]
    
    best_x = particles[0,best_particles_idx][0:100]
    best_weights = weights[best_particles_idx][:100]
    #print(best_x)
    best_y = particles[1,best_particles_idx][0:100]

    p_prom = p_prom[0][new_particles_idx]
    p_prom = p_prom[np.newaxis, :]

    rho_vals = [np.sqrt(best_x[i]**2+best_y[i]**2) for i in range(100)]
    theta_vals = [np.arctan2(best_y[i],best_x[i]) for i in range(100)]
    mean_rho[0][i] = np.mean(rho_vals)
    mean_theta[0][i] = np.mean(theta_vals)
    rho_std_iter[0][i] = np.std(rho_vals)
    theta_std_iter[0][i] = np.std(theta_vals)

    true_x[0][i] = rho_curr*np.cos(theta_curr)
    true_y[0][i] = rho_curr*np.sin(theta_curr)
    est_x[0][i] = np.sum(particles[0,:]*weights)
    #est_x[0][i] = np.mean(best_x[0])
    est_y[0][i] = np.sum(particles[1,:]*weights)
    particles = new_particles
 
    if i in [1, 2]:
        plt.figure()
        plt.grid(True)
        plt.scatter(best_x, best_y, s=200 * best_weights, c='r')
        plt.plot(rho_curr * np.cos(theta_curr), rho_curr * np.sin(theta_curr), 'b+', markersize=10)
        plt.title(f'Najbolje cestice za i={i}')
        plt.xlabel('X ')
        plt.ylabel('Y ')
        plt.legend(['Merenje', 'Najbolje čestice'])
        plt.show()

# Plot for rho
plt.figure()
plt.grid(True)
plt.plot(range(1, len(rho) + 1), rho, 'r+', markersize=10, linewidth=2)
plt.plot(range(1, len(rho) + 1), mean_rho[0], 'bo-', linewidth=2)
for i in range(len(rho)):
    plt.plot([i + 1, i + 1], [mean_rho[0][i] - 2 * rho_std_iter[0][i], mean_rho[0][i] + 2 * rho_std_iter[0][i]], 'g', linewidth=2)
plt.title('Poređenje merenja i srednjih vrednosti za ro')
plt.xlabel('Vreme (sekunde)')
plt.ylabel('ro')
plt.legend(['Merenje', 'Estimacija', 'Interval poverenja (2σ)'], loc='best')
plt.show()

# Plot for theta
plt.figure()
plt.grid(True)
plt.plot(range(1, len(theta) + 1), np.array(theta), 'r+', markersize=10, linewidth=2)
plt.plot(range(1, len(theta) + 1),  mean_theta[0] , 'bo-', linewidth=2)
for t in range(len(theta)):
    plt.plot([t + 1, t + 1],  np.array([mean_theta[0][t] - 2 * theta_std_iter[0][t], mean_theta[0][t] + 2 * theta_std_iter[0][t]]), 'g', linewidth=2)
plt.title('Poređenje merenja i srednjih vrednosti za teta')
plt.xlabel('Vreme[s])')
plt.ylabel('teta[rad])')
plt.legend(['Merenje', 'Estimacija', 'Interval poverenja (2σ)'], loc='best')
plt.show()

#%Plot for paths
plt.figure()
plt.grid(True)
plt.plot(true_x[0], true_y[0], 'r-', linewidth=2)
plt.plot(est_x[0], est_y[0], 'b-', linewidth=2)
plt.title('Poređenje prave putanje i procenjene putanje')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Prava putanja', 'Procenjena putanja'], loc='best')
plt.show()
