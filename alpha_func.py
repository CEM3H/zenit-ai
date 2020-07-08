# функция расчета IV, GINI и logloss для категориальных переменных с корректировкой целевой по alpha
from sklearn.metrics import log_loss


def cat_features_alpha_logloss(df, predictor, target, alpha, seed = 100, plot_i = False):
    """
    функция расчета IV, GINI и logloss для категориальных переменных с корректировкой целевой по alpha
    
    
    """
    
    df[predictor] = df[predictor].fillna('NO_INFO')
    L_logloss_mean = []
    GINI_IV_mean = []
    for alpha_i in tqdm(alpha):
        logloss_i = []
        GINI_i = []
        IV_i = []
        for seed_i in range(seed):
            X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=target), df[target], 
                                                    test_size=0.3, random_state=seed_i, stratify=df[target])
            X_train[target] = y_train
            X_test[target] = y_test
            X_test = X_test[[predictor, target]]
            X_test_WOE = pd.DataFrame()
            X_test_WOE['Target'] = X_test[target]
                       
            tmp = pd.crosstab(X_train[predictor], X_train[target], normalize=False, margins='index')
            tmp.rename(columns={'All':'Values', 1:'Target_cnt'}, inplace=True)
            tmp['Target'] = tmp['Target_cnt'] / tmp['Values']
            tmp['Non Target'] = 1 - tmp['Target']
            tmp = tmp[['Non Target', 'Target', 'Values', 'Target_cnt']].drop('All').reset_index()
            tmp.columns = tmp.columns.rename('')

            
            # расчет оптимальной целевой для группы, формула и детали в видео
            # https://www.youtube.com/watch?v=g335THJxkto&list=PLLIunAIxCvT8ZYpC6-X7H0QfAQO9H0f-8&index=12&t=0s
            # pd = (y_local * K + Y_global * alpha) / (K + alpha)
            Y_global = y_train.mean()
            K = tmp['Values'] / tmp['Values'].sum()
            tmp['Target_transformed'] = (tmp['Target'] * K + Y_global * alpha_i) / (K + alpha_i)
            tmp['Target_cnt_transformed'] = np.floor(tmp['Values'] * tmp['Target_transformed']).astype(int)

            # если пустых значений = 1 - необходимо добавить в таблицу это значение
            if 'NO_INFO' not in tmp[predictor].values:
                tmp = tmp.append({predictor : 'NO_INFO',
                                'Non Target' : df[(df[predictor] == 'NO_INFO') & (df[target] == 0)].shape[0],
                                'Target' : df[(df[predictor] == 'NO_INFO') & (df[target] == 1)].shape[0],
                                'Values' : df[(df[predictor] == 'NO_INFO')].shape[0],
                                'Target_cnt' : df[(df[predictor] == 'NO_INFO') & (df[target] == 1)].shape[0],
                                'Target_transformed' : X_train[target].mean(),
                                'Target_cnt_transformed' : (df[(df[predictor] == 'NO_INFO')].shape[0]) * X_train[target].mean()
                               }, ignore_index=True)

            tmp.sort_values(by = 'Values', inplace=True, ascending=False)
            tmp = tmp.reset_index(drop=True)
            order = list(tmp[predictor])

            # расчет WOE и IV на Train
            df_i = tmp[['Values', 'Target_cnt_transformed', predictor]]
            df_i.rename(columns={'Values' : 'sample_count', 
                                 'Target_cnt_transformed' : 'target_count',
                                  predictor : 'groups'}, inplace=True)
            WOE_i = IVWOE(df_i)
            return WOE_i

            # задаем промежуточную функцию для WOE преобразования переменной из исходного датафрейма 
            # по рассчитанным WOE из IVWOE
            def calc_woe_i(row_value):
                if row_value not in WOE_i['groups']:
                    return 0
                else:
                    i = 0
                    while row_value not in WOE_i['groups'][i]: i += 1
                    return WOE_i['WOE'][i]

            X_test_WOE['WOE'] = X_test[predictor].apply(calc_woe_i)
            roc_auc_i = roc_auc_score(X_test_WOE['Target'], X_test_WOE['WOE'])
            
            X_test = pd.merge(X_test, tmp[[predictor, 'Target_transformed']], how='left', on=predictor)
            #print(X_test[X_test['Target_transformed'].isna()])
            
#             print(seed_i)
#             print(X_test['Target_transformed'].isnull().sum())
#             print(X_test['Target_transformed'].loc[X_test['Target_transformed'].isnull()])
#             print(np.isinf(X_test['Target_transformed']).sum())
            
#             logloss_i.append(log_loss(X_test[target], X_test['Target_transformed']))
            logloss_i.append(log_loss(X_test[target], X_test['Target_transformed'].fillna(0)))
            IV_i.append(WOE_i['IV'].sum())
            GINI_i.append(abs(2 * roc_auc_i - 1))
            
        L_logloss_mean.append([alpha_i, np.mean(logloss_i)])
        GINI_IV_mean.append([alpha_i, np.mean(GINI_i), np.mean(IV_i)])
        
    df_cat_features_alpha_GINI_IV = pd.DataFrame(GINI_IV_mean, columns=['alpha', 'GINI', 'IV'])
    
    df_cat_features_alpha_logloss = pd.DataFrame(L_logloss_mean, columns=['alpha', 'logloss'])
    logloss_min = df_cat_features_alpha_logloss['logloss'].min()
    alpha_opt = df_cat_features_alpha_logloss[df_cat_features_alpha_logloss['logloss'] == logloss_min]['alpha'].values[0]
    
    print('feature =', predictor)
    print('log loss min =', logloss_min)
    print('alpha optimum =', alpha_opt)
    
    if plot_i:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        ax.plot(df_cat_features_alpha_logloss['alpha'], df_cat_features_alpha_logloss['logloss'], label = 'logloss_test', marker='o', ms = 3, color = 'red')
        ax2 = ax.twinx()
        ax2.plot(df_cat_features_alpha_GINI_IV['alpha'], df_cat_features_alpha_GINI_IV['IV'], label = 'IV_train', marker='o', ms = 3, color = 'blue')
        ax2.plot(df_cat_features_alpha_GINI_IV['alpha'], df_cat_features_alpha_GINI_IV['GINI'], label = 'GINI_test', marker='o', ms = 3, color = 'green')
        
        ax_y_step = (max(df_cat_features_alpha_logloss['logloss']) - min(df_cat_features_alpha_logloss['logloss'])) * 0.1
        ax_y_min = min(df_cat_features_alpha_logloss['logloss']) - ax_y_step
        ax_y_max = max(df_cat_features_alpha_logloss['logloss']) + ax_y_step
        ax.set_ylim(ax_y_min, ax_y_max)
        
        ax2_y_step = (max(max(df_cat_features_alpha_GINI_IV['IV']), max(df_cat_features_alpha_GINI_IV['GINI'])) - min(min(df_cat_features_alpha_GINI_IV['IV']), min(df_cat_features_alpha_GINI_IV['GINI']))) * 0.1
        ax2_y_min = min(min(df_cat_features_alpha_GINI_IV['IV']), min(df_cat_features_alpha_GINI_IV['GINI'])) - ax2_y_step
        ax2_y_max = max(max(df_cat_features_alpha_GINI_IV['IV']), max(df_cat_features_alpha_GINI_IV['GINI'])) + ax2_y_step
        ax2.set_ylim(ax2_y_min, ax2_y_max)
        
        ax.tick_params(axis="x", labelsize=12)
        ax2.tick_params(axis="x", labelsize=12)
        ax.set_xlabel('alpha', fontsize=16)
        ax.set_ylabel('logloss', fontsize=16)
        ax2.set_ylabel('GINI and IV', fontsize=16)
        ax.legend(loc = "upper left")
        ax2.legend(loc = "upper right")
        plt.grid(True)
        plt.title('Распределение logloss, GINI и IV от значения alpha', fontsize=20)
        plt.show()
    
    return(alpha_opt)






def _regularize_groups(self, X, y, alpha_list, n_seeds=100):
        L_logloss_mean = []
        GINI_IV_mean = []
        for alpha in alpha_list:
            logloss_i = []
            IV_i = []
            GINI_i = []
            for seed in range(n_seeds):
                tmp = self._regularize_single(X, y, alpha, seed)
                logloss_i.append(tmp[0])
                IV_i.append(tmp[1])
                GINI_i.append(tmp[2])
            L_logloss_mean.append([alpha, np.mean(logloss_i)])
            GINI_IV_mean.append([alpha, np.mean(GINI_i), np.mean(IV_i)])

        df_cat_features_alpha_GINI_IV = pd.DataFrame(GINI_IV_mean, columns=['alpha', 'GINI', 'IV'])
    
        df_cat_features_alpha_logloss = pd.DataFrame(L_logloss_mean, columns=['alpha', 'logloss'])
        logloss_min = df_cat_features_alpha_logloss['logloss'].min()
        alpha_opt = df_cat_features_alpha_logloss[df_cat_features_alpha_logloss['logloss'] == logloss_min]['alpha'].values[0]
    
        print('feature =', X.name)
        print('log loss min =', logloss_min)
        print('alpha optimum =', alpha_opt)

    def _regularize_single(self, X, y, alpha, seed):
        """
        X - датафрейм из одного предиктора
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3, 
                                                            random_state=seed, 
                                                            stratify=y)
        tmp = self._grouping(X_train, y_train, alpha)
        tmp_stats = self._statistic(tmp)
        
        def calc_woe_i(row_value):
                if row_value not in WOE_i['groups']:
                    return 0
                else:
                    i = 0
                    while row_value not in WOE_i['groups'][i]: i += 1
                    return WOE_i['WOE'][i]

        X_test_woe = X_test[predictor].apply(calc_woe_i)
        roc_auc_i = roc_auc_score(y_test, X_test_woe)

        woe_map = dict(zip(tmp_stats.stats['groups'], tmp_stats.stats['WOE']))
        target_rate_transformed = X_test.replace(woe_map)
        target_rate_transformed.loc[~target_rate_transformed.isin(woe_map.keys())] = 0
        roc_auc_i = roc_auc_score(y_test, target_rate_transformed)

        logloss = (log_loss(y_test, target_rate_transformed.fillna(0)))
        IV = (tmp_stats['IV'].sum())
        gini = (abs(2 * roc_auc_i - 1))

        return logloss, IV, gini
        

        
