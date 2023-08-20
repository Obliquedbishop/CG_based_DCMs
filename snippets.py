# Sample code for basic grid search hyperparameter tuning


lambda_vals = [0.001, 0.01, 0.1, 0, 10, 100, 1000]
f2_val_scores = []
params_list = []
best_l1_l2 = None
min_f2_score = 2

ind_spec_vars = ['AGE', 'Female', 'INCOME', 'nearest_bus_dist', 'nearest_train_dist',]
params = initiate_params(ind_spec_vars_list=ind_spec_vars)
ohe_true_y = tf.one_hot(y_val-1, 6, on_value=1, off_value=0, axis=-1, dtype='float32')
avail_vals = generate_avail_data(X_val)
x_names = X_val.columns
x_vals = X_val.values


for l1 in lambda_vals:
    for l2 in lambda_vals:
        loss_grad_func = loss_val_grad_factory(l1, l2)
        bfgs_func = mnl_with_bfgs_factory(loss_grad_func)
        results = run(bfgs_func)
        
        y_pred_val = predict(X_val.columns, X_val.values, results.position, 
                 avail_vals, ind_spec_vars_list=ind_spec_vars)[1].numpy() + 1
        
        f2_val = fbeta_score(y_val, y_pred_val, beta=2, average='macro')
        
        f2_val_scores.append(f2_val)
        params_list.append(results.position)
        
        if f2_val < min_f2_score:
            min_f2_score = f2_val
            best_l1_l2 = (l1, l2)
        
        print(l1, l2, f2_val)
        
        
# Factory for functions
def mnl_with_bfgs_factory(loss_grad_func):
    @tf.function
    def mnl_with_bfgs():
        return tfp.optimizer.bfgs_minimize(loss_grad_func, 
                                           initial_position=tf.reshape(params, shape=[-1]),
                                          tolerance=1e-8,
                                          max_iterations=500)
    return mnl_with_bfgs

def loss_val_grad_factory(l1, l2):
    @make_val_and_grad_fn
    def loss_val_nd_loss_grad(params):
        curr_utils = utilities(x_names, x_vals, params, ind_spec_vars_list=ind_spec_vars)
        curr_loss = loss(ohe_true_y, curr_utils, avail_vals)
        penalty_loss = apply_regularization(curr_loss, params, l1_lambda=l1, l2_lambda=l2)
        return penalty_loss
    return loss_val_nd_loss_grad

