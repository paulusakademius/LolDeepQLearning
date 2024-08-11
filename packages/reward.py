

cs=0
kills=0
deaths=0
gold=0

def getReward(state): #determines the reward for given state
    old_cs = cs
    cs = getCs()
    cs_reward = (cs - old_cs)*3

    old_kills = kills
    kills = getKills()
    kills_reward = (kills - old_kills) * 5

    old_gold = gold
    gold = getGold()
    gold_reward = (gold - old_gold) * 0.1

    old_deaths = deaths
    deaths = getDeaths()
    death_punishment = old_deaths - deaths

    return cs_reward + kills_reward + gold_reward + death_punishment