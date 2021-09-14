from sklearn.metrics import confusion_matrix
import math
import numpy as np

NOISE_LEVELS = ['-3']
def far(net, data, size_limit = 0, frr = 1, far=1, plot = True):
    
    # 1.Расчитываем конфьюжн матрицу
    # 2.Считаем tn, fp, fn, tp 
    # 3.По ним на каждом threshold вычисляем far, frr
    # 4.Возвращаем нужные значения в зависимости от условий на far, frr 
    
    # В зависимости от трешхолда вычисляем предскзания по классам
    def apply_threshold(y_score, t = 0.5):
        return [1 if y >= t else 0 for idx, y in enumerate(y_score)]
    
    def fix_frr(y_true, y_score, frr_target):        
        t = 1e-9        
        # Расчитывваем FAR для FRR = 1%
        while t < 1.0:            
            tn, fp, fn, tp = confusion_matrix(y_true, apply_threshold(y_score, t)).ravel()
            
            far = (fp * 100) / (fp + tn)
            frr = (fn * 100) / (fn + tp)
            
            if frr >= frr_target:
                return far, frr, t
            
            t *= 1.1       
        return far, frr, t
    
    def fix_far(y_true, y_score, far_target):       
        t = 1e-9
        
        # Расчитывваем FRR для FAR = 1%
        while t < 1.0:
            
            tn, fp, fn, tp = confusion_matrix(y_true, apply_threshold(y_score, t)).ravel()
            
            far = (fp * 100) / (fp + tn)
            frr = (fn * 100) / (fn + tp)
            
            if int(far) <= far_target:
                return frr, far, t
            
            t *= 1.1       
        return frr, far, t

    def fix_eer(y_true, y_score):       
        t = 1e-9
        
        # Расчитываем FRR для FAR = 1%
        while t < 1.0:
            
            tn, fp, fn, tp = confusion_matrix(y_true, apply_threshold(y_score, t)).ravel()
            
            far = (fp * 100) / (fp + tn)
            frr = (fn * 100) / (fn + tp)
            
            if int(far) == int(frr):
                return far, frr, t
            
            t *= 1.1       
        

    print('Network metrics:')
    
    for lvl in NOISE_LEVELS:        
        
        y_true, y_score = test_predict(net, data, size_limit, lvl)
        
        print('FA: %0.2f%% for fixed FR at %0.2f%%, tr %0.2f%%' % fix_frr(y_true, y_score, frr))
        print('FR: %0.2f%% for fixed FA at %0.2f%%, tr %0.2f%%' % fix_far(y_true, y_score, far))
        print('EER: %0.2f%% for FA = FR=%0.2f%%, tr %0.2f%%' % fix_eer(y_true, y_score))


