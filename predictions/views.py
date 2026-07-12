import os
import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from xgboost import XGBClassifier

# Chargement global des ressources (une seule fois au démarrage)
BASE_DIR = settings.BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, 'ia_models', 'best_xgb_model.json')
TEAM_STATS_PATH = os.path.join(BASE_DIR, 'ia_models', 'team_stats.pkl')
DF_RANKING_PATH = os.path.join(BASE_DIR, 'ia_models', 'df_ranking.pkl')

TEAM_STATS = joblib.load(TEAM_STATS_PATH)
DF_RANKING = joblib.load(DF_RANKING_PATH)

MODEL = XGBClassifier()
MODEL.load_model(MODEL_PATH)

# ====================== FONCTIONS UTILITAIRES ======================

def get_last_ranking():
    """Retourne le classement le plus récent"""
    return DF_RANKING.sort_values('rank_date').groupby('country_full').last().reset_index()

def predire_match_brut(eq_a, eq_b, rang_a, rang_b):
    """Prédiction brute d'un match"""
    stats_a = TEAM_STATS.get(eq_a, {'rolling_scored': 1.0, 'rolling_conceded': 1.0, 
                                   'rolling_points': 1.0, 'fifa_points': 0, 
                                   'rank_change': 0, 'points_change': 0})
    stats_b = TEAM_STATS.get(eq_b, {'rolling_scored': 1.0, 'rolling_conceded': 1.0, 
                                   'rolling_points': 1.0, 'fifa_points': 0, 
                                   'rank_change': 0, 'points_change': 0})
    
    features = pd.DataFrame([{
        'is_home_country': 0,
        'tournament_weight': 4,
        'home_rolling_scored': stats_a['rolling_scored'],
        'home_rolling_conceded': stats_a['rolling_conceded'],
        'home_rolling_points': stats_a['rolling_points'],
        'away_rolling_scored': stats_b['rolling_scored'],
        'away_rolling_conceded': stats_b['rolling_conceded'],
        'away_rolling_points': stats_b['rolling_points'],
        'home_rank': rang_a,
        'home_fifa_points': stats_a['fifa_points'],
        'home_rank_change': stats_a['rank_change'],
        'home_points_change': stats_a['points_change'],
        'away_rank': rang_b,
        'away_fifa_points': stats_b['fifa_points'],
        'away_rank_change': stats_b['rank_change'],
        'away_points_change': stats_b['points_change'],
        'rank_difference': rang_a - rang_b,
        'points_difference': float(stats_a['fifa_points']) - float(stats_b['fifa_points']),
        'goals_scored_difference': stats_a['rolling_scored'] - stats_b['rolling_scored'],
        'goals_conceded_difference': stats_a['rolling_conceded'] - stats_b['rolling_conceded']
    }])
    
    probs = MODEL.predict_proba(features)
    # Retourne la probabilité que l'équipe A gagne (ajusté selon tes classes)
    return float(probs[0][2] + (probs[0][1] * 0.5))  # À adapter si tes classes sont différentes

# ====================== VIEWS ======================

def dashboard_favoris(request):
    last_ranking = get_last_ranking()
    top_16 = last_ranking.sort_values('rank').head(16).copy()
    
    pays = list(top_16['country_full'])
    scores = {p: 0 for p in pays}
    
    # Mini tournoi round-robin simplifié
    for i in range(len(pays)):
        for j in range(i + 1, len(pays)):
            pa, pb = pays[i], pays[j]
            r_a = int(top_16[top_16['country_full'] == pa]['rank'].iloc[0])
            r_b = int(top_16[top_16['country_full'] == pb]['rank'].iloc[0])
            
            if predire_match_brut(pa, pb, r_a, r_b) > 0.5:
                scores[pa] += 3
            else:
                scores[pb] += 3
    
    # Classement final
    favs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_favoris = [
        {
            'index': idx,
            'pays': p,
            'score': sc,
            'rang_actuel': int(top_16[top_16['country_full'] == p]['rank'].iloc[0])
        }
        for idx, (p, sc) in enumerate(favs[:10], 1)
    ]
    
    return render(request, 'predictions/dashboard.html', {
        'top_favoris': top_favoris,
        'top_16': top_16.to_dict('records')
    })


def match_predictor(request):
    last_ranking = get_last_ranking()
    liste_pays = sorted(last_ranking['country_full'].unique())
    
    context = {'liste_pays': liste_pays, 'resultat': None}
    
    if request.method == "POST":
        eq_a = request.POST.get('equipe_a')
        eq_b = request.POST.get('equipe_b')
        
        if eq_a and eq_b and eq_a != eq_b:
            r_a = int(last_ranking[last_ranking['country_full'] == eq_a]['rank'].iloc[0])
            r_b = int(last_ranking[last_ranking['country_full'] == eq_b]['rank'].iloc[0])
            
            p_a = predire_match_brut(eq_a, eq_b, r_a, r_b) * 100
            context['resultat'] = {
                'eq_a': eq_a,
                'eq_b': eq_b,
                'pct_a': round(p_a, 1),
                'pct_b': round(100 - p_a, 1)
            }
    
    return render(request, 'predictions/confrontation.html', context)


def page_scenarios(request):
    last_ranking = get_last_ranking()
    
    # Préparation de la liste avec les rangs pour le JavaScript
    liste_pays = last_ranking[['country_full', 'rank']].copy()
    liste_pays = liste_pays.rename(columns={'country_full': 'name'}).to_dict('records')
    
    context = {
        'liste_pays': liste_pays, 
        'resultat_scenario': None
    }
    
    if request.method == "POST":
        eq_a = request.POST.get('scen_equipe_a')
        eq_b = request.POST.get('scen_equipe_b')
        bonus_a = int(request.POST.get('bonus_a', 0))
        malus_b = int(request.POST.get('malus_b', 0))
        
        if eq_a and eq_b and eq_a != eq_b:
            # Récupération des rangs réels
            r_a_base = int(last_ranking[last_ranking['country_full'] == eq_a]['rank'].iloc[0])
            r_b_base = int(last_ranking[last_ranking['country_full'] == eq_b]['rank'].iloc[0])
            
            r_a_futur = max(1, r_a_base - bonus_a)
            r_b_futur = r_b_base + malus_b
            
            p_a = predire_match_brut(eq_a, eq_b, r_a_futur, r_b_futur)
            
            # Simulation Monte Carlo
            tirages = np.random.choice([eq_a, eq_b], size=5000, p=[p_a, 1 - p_a])
            pct_a = (np.sum(tirages == eq_a) / 5000) * 100
            
            context['resultat_scenario'] = {
                'eq_a': eq_a,
                'eq_b': eq_b,
                'pct_a': round(pct_a, 1),
                'pct_b': round(100 - pct_a, 1),
                'rang_a': r_a_futur,
                'rang_b': r_b_futur,
                'vainqueur': eq_a if pct_a > 50 else eq_b
            }
    
    return render(request, 'predictions/scenarios.html', context)