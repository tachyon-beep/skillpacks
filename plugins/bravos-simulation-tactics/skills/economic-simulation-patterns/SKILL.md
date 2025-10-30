---
name: economic-simulation-patterns
description: Master game economies - faucets, sinks, arbitrage prevention, pricing
---

# Economic Simulation Patterns

## Description
Master supply/demand dynamics, market simulation, production chains, and economic balance for game economies. Apply faucet/sink analysis, exploit validation, and price stabilization to create stable, engaging economies that resist inflation, arbitrage loops, and market manipulation over months of gameplay.

## When to Use This Skill
Use this skill when implementing or debugging:
- Trading games with dynamic markets (space sims, MMO auction houses)
- Resource-based economies (crafting, production chains)
- Player-to-player marketplaces
- NPC vendors and dynamic pricing
- Currency systems and money supply management
- Multi-month economic balance and stability

Do NOT use this skill for:
- Simple fixed-price vendors (no supply/demand)
- Single-player games without trading
- Abstract resources without markets (XP, skill points)
- Cosmetic-only economies (no gameplay impact)

---

## Quick Start (Time-Constrained Implementation)

If you need a working economy quickly (< 4 hours), follow this priority order:

**CRITICAL (Never Skip)**:
1. **Faucets < Sinks**: Money entering system must be less than money leaving
2. **Validate chains**: Production chains must be net-negative vs NPCs (no infinite money)
3. **Price bounds**: Set min/max prices (prevent 0 or infinity)
4. **NPC bid-ask spread**: NPCs buy at 50-70% of sell price

**IMPORTANT (Strongly Recommended)**:
5. Transaction fees (1-5% per trade removes money from system)
6. Consumables (ammo, fuel, food - require constant spending)
7. Price dampening (1.01x adjustments, not 1.1x)
8. Daily limits (prevent 24/7 bot farming)

**CAN DEFER** (Optimize Later):
- Complex market makers
- Dynamic recipe costs
- Regional price variations
- Advanced anti-manipulation

**Example - Minimal Stable Economy in 2 Hours**:
```python
class MinimalEconomy:
    def __init__(self):
        # Prices with bounds
        self.prices = {'Iron': 10}
        self.MIN_PRICE = 1
        self.MAX_PRICE = 100

    def sell_to_npc(self, player, item, qty):
        # NPC buys at 60% of market price (bid-ask spread)
        revenue = self.prices[item] * 0.6 * qty
        player.credits += revenue * 0.95  # 5% transaction fee (SINK)

    def buy_from_npc(self, player, item, qty):
        cost = self.prices[item] * qty
        if player.credits >= cost:
            player.credits -= cost
            return True
        return False

    def update_price(self, item, demand, supply):
        # Dampened adjustment (1.01x not 1.1x)
        if demand > supply:
            self.prices[item] *= 1.01
        else:
            self.prices[item] *= 0.99

        # Enforce bounds
        self.prices[item] = clamp(self.prices[item], self.MIN_PRICE, self.MAX_PRICE)

    def validate_recipe(self, recipe):
        # CRITICAL: Verify no infinite money
        input_cost = sum(self.prices[mat] * qty for mat, qty in recipe.inputs.items())
        output_value = self.prices[recipe.output] * 0.6  # NPC buy price

        assert output_value < input_cost, f"Recipe {recipe.output} creates infinite money!"
```

This gives you:
- Money sinks (transaction fees)
- No arbitrage (bid-ask spread + validation)
- Stable prices (bounds + dampening)

Refine later based on playtesting.

---

## Core Concepts

### 1. Faucets and Sinks (Money Supply Control)

Every game economy has **faucets** (money entering) and **sinks** (money leaving). The balance determines long-term stability.

**Faucets** (Money Creation):
```python
# Examples of money entering the economy
class MoneyFaucets:
    def new_player_bonus(self, player):
        player.credits += 10000  # +10k per new player

    def quest_reward(self, player, quest):
        player.credits += 5000  # +5k per quest

    def npc_mission(self, player):
        player.credits += 1000  # +1k per mission

    def monster_drops(self, player, monster):
        player.credits += 50  # +50 per kill

    def calculate_daily_faucet(self, player_count):
        # Total money entering per day
        new_players = 100 * 10000          # 1,000,000
        quests = player_count * 5 * 5000   # 5 quests/day
        missions = player_count * 10 * 1000  # 10 missions/day
        monsters = player_count * 100 * 50   # 100 kills/day

        return new_players + quests + missions + monsters
```

**Sinks** (Money Destruction):
```python
# Examples of money leaving the economy
class MoneySinks:
    def transaction_fee(self, trade_value):
        # 2% of every trade removed from game
        fee = trade_value * 0.02
        # Credits disappear (not given to anyone)
        return -fee

    def repair_cost(self, player, item):
        # Items degrade, require repairs
        cost = item.max_value * 0.1
        player.credits -= cost

    def consumable_purchase(self, player):
        # Fuel, ammo, food - constant spending
        player.credits -= 100  # Must buy from NPC (sinks)

    def luxury_items(self, player):
        # Cosmetics, housing - pure sinks
        player.credits -= 50000

    def market_listing_fee(self, listing):
        # Fee to list item on marketplace
        return listing.value * 0.05

    def calculate_daily_sink(self, player_count):
        # Total money leaving per day
        fees = player_count * 20 * 1000 * 0.02     # 20 trades/day
        repairs = player_count * 5 * 500            # 5 repairs/day
        consumables = player_count * 100            # Daily consumables
        luxuries = player_count * 0.1 * 50000       # 10% buy luxury/day

        return fees + repairs + consumables + luxuries
```

**The Golden Rule**:
```python
def validate_economy_stability(faucets, sinks, player_count):
    '''CRITICAL: Sinks must exceed faucets for stability'''
    daily_faucet = calculate_daily_faucet(player_count)
    daily_sink = calculate_daily_sink(player_count)

    ratio = daily_sink / daily_faucet

    if ratio < 0.8:
        print("CRITICAL: Runaway inflation! Sinks too weak.")
        print(f"  Faucet: {daily_faucet}, Sink: {daily_sink}")
        print(f"  Ratio: {ratio:.2f} (need > 0.8)")
        return False

    elif ratio > 1.2:
        print("WARNING: Deflation! Sinks too strong.")
        print(f"  Players will run out of money.")
        return False

    else:
        print(f"HEALTHY: Sink/Faucet ratio = {ratio:.2f}")
        return True
```

**Real-World Target**:
- **0.8-1.0**: Slight inflation (prices gradually rise)
- **1.0-1.1**: Stable (target for most games)
- **1.1-1.3**: Slight deflation (late-game sink)

**Why This Matters**:
- Ratio < 0.8: Money supply grows exponentially → hyperinflation
- Ratio > 1.3: Money supply shrinks → players can't afford anything
- Ratio 0.9-1.1: Stable economy for months/years

### 2. Arbitrage Prevention (No Free Money)

**Arbitrage**: Buying low and selling high for guaranteed profit with no risk.

**The Problem**:
```python
# ❌ BROKEN: Infinite money exploit
class BrokenEconomy:
    def __init__(self):
        self.npc_sell_price = 10  # NPC sells Iron for 10
        self.npc_buy_price = 10   # NPC buys Iron for 10 (SAME!)

    def exploit(self, player):
        while True:
            # Buy from NPC
            player.credits -= 10
            player.inventory['Iron'] += 1

            # Sell to NPC
            player.inventory['Iron'] -= 1
            player.credits += 10

            # Net: 0 profit (but with manufacturing...)

            # Craft: 50 Iron → 1 Hull (takes time but guaranteed)
            if player.inventory['Iron'] >= 50:
                player.inventory['Iron'] -= 50
                player.inventory['Hull'] += 1

                # Sell Hull to NPC
                player.inventory['Hull'] -= 1
                player.credits += 1000  # Hull sells for 1000

                # Cost: 50 * 10 = 500 credits
                # Revenue: 1000 credits
                # Profit: 500 credits (100% guaranteed!)
```

**The Fix: Bid-Ask Spread**:
```python
# ✅ SAFE: NPCs buy low, sell high
class SafeEconomy:
    def __init__(self):
        self.market_price = 10  # "Fair" price

        # NPCs sell at premium
        self.npc_sell_price = self.market_price * 1.0  # 10 (sell to players)

        # NPCs buy at discount
        self.npc_buy_price = self.market_price * 0.6   # 6 (buy from players)

    def validate_no_arbitrage(self):
        '''Verify all production chains are net-negative'''
        recipes = {
            'Hull': {'Iron': 50, 'Silicon': 20},
            'Electronics': {'Silicon': 30, 'Platinum': 10},
        }

        for output, inputs in recipes.items():
            # Calculate cost (buying materials from NPC)
            input_cost = sum(
                self.npc_sell_price[mat] * qty
                for mat, qty in inputs.items()
            )

            # Calculate revenue (selling product to NPC)
            output_revenue = self.npc_buy_price[output]

            profit = output_revenue - input_cost

            if profit > 0:
                print(f"ERROR: {output} creates infinite money!")
                print(f"  Cost: {input_cost}, Revenue: {output_revenue}")
                print(f"  Profit: {profit} (should be negative!)")
                raise ValueError(f"Recipe {output} is exploitable!")

            else:
                print(f"SAFE: {output} requires player trading for profit")
```

**Why This Works**:
- Players MUST trade with each other to profit
- NPC loops are always net-negative
- Crafting is only profitable if you sell to players (creates real economy)

**Bid-Ask Spread Guidelines**:
- **Conservative**: NPC buys at 50% (safe, forces player trading)
- **Moderate**: NPC buys at 60-70% (reasonable, still safe)
- **Aggressive**: NPC buys at 80% (risky, verify carefully!)
- **Never**: NPC buys at 100% (guaranteed exploits)

### 3. Supply and Demand Dynamics

**Basic Model**:
```python
class SupplyDemandMarket:
    def __init__(self):
        self.price = 10
        self.supply = 0     # Items available for sale
        self.demand = 0     # Items players want to buy

    def update_price_simple(self):
        '''Simple supply/demand adjustment'''
        if self.demand > self.supply:
            # Shortage: Price rises
            self.price *= 1.05
        elif self.supply > self.demand:
            # Surplus: Price falls
            self.price *= 0.95
```

**Problem**: This is unstable! Prices oscillate wildly.

**Better: Dampened Adjustment**:
```python
    def update_price_dampened(self):
        '''Dampened adjustment for stability'''
        # Calculate imbalance
        if self.supply > 0:
            ratio = self.demand / self.supply
        else:
            ratio = 10.0  # High demand, no supply

        # Dampened adjustment (smaller steps)
        if ratio > 1.1:  # Demand > Supply by 10%+
            self.price *= 1.01  # Increase 1% (not 5%!)
        elif ratio < 0.9:  # Supply > Demand by 10%+
            self.price *= 0.99  # Decrease 1%

        # Reset counters for next period
        self.supply = 0
        self.demand = 0
```

**Best: Exponential Moving Average**:
```python
    def update_price_ema(self, recent_trades):
        '''Smooth price adjustment using EMA'''
        # Calculate average trade price from recent trades
        if len(recent_trades) == 0:
            return

        avg_trade_price = sum(t.price for t in recent_trades) / len(recent_trades)

        # Exponential moving average (smooth towards trade price)
        SMOOTHING = 0.1  # 10% weight to new data
        self.price = self.price * (1 - SMOOTHING) + avg_trade_price * SMOOTHING

        # Still enforce bounds
        self.price = clamp(self.price, self.MIN_PRICE, self.MAX_PRICE)
```

**Advanced: Market Clearing Price**:
```python
class MarketClearingPrice:
    def __init__(self):
        self.buy_orders = []   # [(price, quantity), ...]
        self.sell_orders = []  # [(price, quantity), ...]

    def find_clearing_price(self):
        '''Find price where supply = demand'''
        # Sort orders
        self.buy_orders.sort(reverse=True)   # Highest price first
        self.sell_orders.sort()              # Lowest price first

        # Walk through order book
        for sell_price, sell_qty in self.sell_orders:
            # Count how many buyers at this price or higher
            demand_at_price = sum(
                qty for price, qty in self.buy_orders
                if price >= sell_price
            )

            # Count how many sellers at this price or lower
            supply_at_price = sum(
                qty for price, qty in self.sell_orders
                if price <= sell_price
            )

            if demand_at_price >= supply_at_price:
                # This is the clearing price!
                return sell_price

        # No clearing price found
        return None
```

**Decision Framework**:
- **Simple games**: Use dampened adjustment (easy, stable enough)
- **Trading-focused games**: Use market clearing price (realistic, complex)
- **Hybrid**: EMA for NPC prices, order book for player marketplace

### 4. Price Bounds and Stabilization

**Never Let Prices Go to Zero or Infinity**:
```python
class BoundedPricing:
    def __init__(self, base_price):
        self.price = base_price
        self.MIN_PRICE = base_price * 0.1   # Never below 10% of base
        self.MAX_PRICE = base_price * 10.0  # Never above 10x base

    def update_price(self, adjustment):
        self.price *= adjustment

        # Enforce bounds (CRITICAL!)
        self.price = clamp(self.price, self.MIN_PRICE, self.MAX_PRICE)

        # Alternative: Asymptotic bounds (smooth approach to limits)
        # self.price = self.MIN_PRICE + (self.MAX_PRICE - self.MIN_PRICE) * (
        #     1 / (1 + math.exp(-(self.price - base_price) / base_price))
        # )
```

**Why Bounds Matter**:
- No bounds: Price can explode to 1,000,000+ or crash to 0.0001
- With bounds: Price stays playable (players can always afford basics)

**NPC Market Makers** (Advanced Stabilization):
```python
class NPCMarketMaker:
    def __init__(self, target_price, liquidity):
        self.target_price = target_price
        self.liquidity = liquidity  # How much NPC will trade

    def provide_liquidity(self, market):
        '''NPC always offers to buy/sell at near-target price'''
        # NPC sells at slight premium
        npc_sell_price = self.target_price * 1.05
        npc_sell_quantity = self.liquidity

        # NPC buys at slight discount
        npc_buy_price = self.target_price * 0.95
        npc_buy_quantity = self.liquidity

        # Add NPC orders to market
        market.add_sell_order(npc_sell_price, npc_sell_quantity, is_npc=True)
        market.add_buy_order(npc_buy_price, npc_buy_quantity, is_npc=True)

    def adjust_target(self, current_price):
        '''NPC slowly adjusts target towards market price'''
        SMOOTHING = 0.01  # Very slow adjustment
        self.target_price = self.target_price * (1 - SMOOTHING) + current_price * SMOOTHING
```

**Effect**:
- Players can always buy/sell (NPC provides liquidity)
- Prices stabilize near target (NPC absorbs shocks)
- Market still responds to player activity (target adjusts slowly)

### 5. Production Chain Balance

**Every recipe must be balanced for profitability and gameplay**:

```python
class ProductionChainBalancer:
    def __init__(self, market):
        self.market = market

    def analyze_chain(self, recipe):
        '''Analyze profitability and balance of a recipe'''
        # Calculate input cost (buying from players at market price)
        input_cost_market = sum(
            self.market.get_price(mat) * qty
            for mat, qty in recipe.inputs.items()
        )

        # Calculate input cost (buying from NPCs)
        input_cost_npc = sum(
            self.market.npc_sell_price(mat) * qty
            for mat, qty in recipe.inputs.items()
        )

        # Calculate output value (selling to players)
        output_value_market = self.market.get_price(recipe.output)

        # Calculate output value (selling to NPCs)
        output_value_npc = self.market.npc_buy_price(recipe.output)

        # Calculate time cost
        time_hours = recipe.crafting_time / 3600  # Convert to hours

        # Profit analysis
        profit_vs_npc = output_value_npc - input_cost_npc
        profit_vs_players = output_value_market - input_cost_market

        profit_per_hour_npc = profit_vs_npc / time_hours
        profit_per_hour_players = profit_vs_players / time_hours

        print(f"\nRecipe: {recipe.output}")
        print(f"  Input cost (NPC): {input_cost_npc:.0f}")
        print(f"  Output value (NPC): {output_value_npc:.0f}")
        print(f"  Profit vs NPC: {profit_vs_npc:.0f} (should be NEGATIVE)")
        print(f"  Profit vs Players: {profit_vs_players:.0f}")
        print(f"  Profit/hour (players): {profit_per_hour_players:.0f}")

        # Validation
        if profit_vs_npc > 0:
            print(f"  ⚠️  ERROR: Creates infinite money vs NPCs!")
            return False

        if profit_vs_players < 0:
            print(f"  ⚠️  WARNING: Unprofitable even with player trading!")

        if profit_per_hour_players < 100:
            print(f"  ⚠️  WARNING: Poor profit/hour (players won't craft)")

        return True

    def balance_all_chains(self, recipes):
        '''Ensure all recipes have similar profit/hour'''
        profits = []

        for recipe in recipes:
            # Calculate profit per hour vs player market
            input_cost = sum(
                self.market.get_price(mat) * qty
                for mat, qty in recipe.inputs.items()
            )
            output_value = self.market.get_price(recipe.output)
            time_hours = recipe.crafting_time / 3600

            profit_per_hour = (output_value - input_cost) / time_hours
            profits.append((recipe.output, profit_per_hour))

        # Sort by profit
        profits.sort(key=lambda x: x[1])

        print("\nProduction Chain Balance:")
        for name, profit in profits:
            print(f"  {name}: {profit:.0f} credits/hour")

        # Check for dominant strategies
        max_profit = profits[-1][1]
        min_profit = profits[0][1]

        if max_profit > min_profit * 3:
            print(f"  ⚠️  WARNING: Imbalanced chains!")
            print(f"  Best: {profits[-1][0]} ({max_profit:.0f}/hr)")
            print(f"  Worst: {profits[0][0]} ({min_profit:.0f}/hr)")
            print(f"  Ratio: {max_profit / min_profit:.1f}x (should be < 3x)")
```

**Balance Guidelines**:
- All chains should have similar profit/hour (within 2-3x)
- No chain should be strictly better (dominant strategy)
- Complex chains can pay more (reward skill/knowledge)
- Late-game chains should be more profitable (progression)

---

## Decision Frameworks

### Framework 1: Economy Complexity Level

**Choose complexity based on game focus**:

| Complexity | Use When | Examples | Implementation Time |
|-----------|----------|----------|---------------------|
| **Simple** | Economy is secondary to core gameplay | Action RPG, FPS | 1-2 days |
| **Moderate** | Trading is important but not central | MMO, Strategy | 1-2 weeks |
| **Complex** | Economy IS the game | Eve Online, Trading sim | 1-3 months |

**Simple Economy**:
```python
# Fixed prices, transaction fees, consumables
class SimpleEconomy:
    PRICES = {'Sword': 100, 'Potion': 10}  # Fixed

    def buy_from_npc(self, player, item):
        player.credits -= self.PRICES[item]

    def sell_to_npc(self, player, item):
        player.credits += self.PRICES[item] * 0.5  # 50% back
```

**Features**:
- Fixed prices (no supply/demand)
- NPC vendors only
- Simple sinks (repair, consumables)
- No player trading

**Use for**: Games where economy is flavor, not focus

**Moderate Economy**:
```python
# Dynamic prices, player marketplace, production
class ModerateEconomy:
    def __init__(self):
        self.prices = {}  # Dynamic
        self.player_market = OrderBook()

    def update_prices(self, trades):
        # EMA based on recent trades
        for item in trades:
            self.prices[item] = ema(self.prices[item], trades[item].avg_price)

    def npc_buy_price(self, item):
        return self.prices[item] * 0.6  # 60% of market

    def npc_sell_price(self, item):
        return self.prices[item] * 1.0  # 100% of market
```

**Features**:
- Dynamic prices (supply/demand)
- Player marketplace (peer trading)
- Production chains (crafting)
- Faucet/sink balance

**Use for**: MMOs, persistent world games

**Complex Economy**:
```python
# Full market simulation, regional prices, contracts
class ComplexEconomy:
    def __init__(self):
        self.regions = {}  # Different prices per region
        self.order_books = {}  # Full order matching
        self.contracts = []  # Player contracts
        self.corporations = []  # Player organizations

    def match_orders(self, item, region):
        '''Full order book matching'''
        book = self.order_books[(item, region)]
        clearing_price = book.find_clearing_price()
        book.execute_trades(clearing_price)

    def transport_goods(self, item, from_region, to_region):
        '''Regional arbitrage (hauling gameplay)'''
        # Different prices in different regions
        # Players profit by hauling goods
        pass
```

**Features**:
- Regional economies (different prices per zone)
- Full order book matching
- Player contracts and corporations
- Hauling/transport gameplay
- Complex production chains

**Use for**: Eve Online-style games, trading simulators

### Framework 2: Player-Driven vs NPC-Driven Markets

**NPC-Driven** (Simpler, More Stable):
```python
class NPCDrivenMarket:
    '''NPCs set prices, players buy/sell from NPCs'''
    def __init__(self):
        self.npc_prices = {}  # NPCs always available

    def buy_from_npc(self, player, item, qty):
        cost = self.npc_prices[item] * qty
        if player.credits >= cost:
            player.credits -= cost
            player.inventory[item] += qty
            return True
        return False

    def sell_to_npc(self, player, item, qty):
        revenue = self.npc_prices[item] * 0.6 * qty  # 60% back
        player.credits += revenue
        player.inventory[item] -= qty
```

**Pros**:
- Always available (no market failures)
- Stable prices (predictable)
- Simple to implement
- Works for small populations

**Cons**:
- Less dynamic
- No emergent gameplay
- Artificial feeling

**Use for**: Single-player, co-op, small multiplayer

**Player-Driven** (Complex, Dynamic):
```python
class PlayerDrivenMarket:
    '''Players set prices, NPCs only provide liquidity'''
    def __init__(self):
        self.order_book = OrderBook()
        self.npc_maker = NPCMarketMaker()  # Backup liquidity

    def create_sell_order(self, seller, item, qty, price):
        '''Player lists item for sale'''
        order = {'seller': seller, 'item': item, 'qty': qty, 'price': price}
        self.order_book.add_sell_order(order)

    def create_buy_order(self, buyer, item, qty, max_price):
        '''Player offers to buy at price'''
        order = {'buyer': buyer, 'item': item, 'qty': qty, 'price': max_price}
        self.order_book.add_buy_order(order)

    def match_orders(self):
        '''Match buy and sell orders'''
        self.order_book.match()

        # If no liquidity, NPC provides backup
        if self.order_book.is_empty():
            self.npc_maker.provide_liquidity(self.order_book)
```

**Pros**:
- Emergent gameplay (market manipulation, speculation)
- Dynamic prices (responds to player behavior)
- Engaging economy (players set prices)
- Scales to large populations

**Cons**:
- Can fail (no buyers/sellers)
- Requires balancing
- Complex implementation
- Needs large player base

**Use for**: MMOs, persistent worlds, large multiplayer

**Hybrid** (Best of Both):
```python
class HybridMarket:
    '''Players trade with each other, NPCs provide fallback'''
    def __init__(self):
        self.player_market = OrderBook()
        self.npc_vendor = NPCVendor()

    def buy_item(self, player, item, qty):
        '''Try player market first, fall back to NPC'''
        # Check player market
        orders = self.player_market.get_sell_orders(item)
        if len(orders) > 0:
            # Buy from cheapest player
            cheapest = min(orders, key=lambda o: o.price)
            if player.credits >= cheapest.price * qty:
                self.execute_player_trade(player, cheapest, qty)
                return True

        # Fall back to NPC (more expensive)
        return self.npc_vendor.buy_from_npc(player, item, qty)
```

**Use for**: Most games (combines stability + dynamics)

### Framework 3: When to Use Regional Economies

**Single Global Market** (Simpler):
- All players see same prices
- No hauling gameplay
- Works for small games

**Regional Markets** (Complex):
- Different prices per zone
- Hauling creates arbitrage opportunities
- Requires large world and population

**Decision Table**:

| Factor | Global Market | Regional Markets |
|--------|---------------|------------------|
| World size | Small (<10 zones) | Large (>20 zones) |
| Player count | <1,000 | >5,000 |
| Travel time | <1 minute | >5 minutes |
| Hauling gameplay | No | Yes (core mechanic) |
| Implementation time | 1 week | 1 month |

**Regional Economy Example**:
```python
class RegionalEconomy:
    def __init__(self):
        self.regions = {
            'Mining Zone': {'Iron': 5, 'Ships': 4000},   # Iron cheap, ships expensive
            'Industrial Zone': {'Iron': 15, 'Ships': 3000},  # Iron expensive, ships cheaper
            'Trade Hub': {'Iron': 10, 'Ships': 3500},   # Average prices
        }

    def get_price(self, region, item):
        return self.regions[region][item]

    def arbitrage_opportunity(self, item):
        '''Find best buy/sell regions for hauling'''
        prices = [(region, self.get_price(region, item)) for region in self.regions]
        prices.sort(key=lambda x: x[1])

        buy_region, buy_price = prices[0]   # Cheapest
        sell_region, sell_price = prices[-1]  # Most expensive

        profit_per_unit = sell_price - buy_price
        print(f"Haul {item} from {buy_region} to {sell_region}")
        print(f"  Profit: {profit_per_unit} per unit")

        return buy_region, sell_region, profit_per_unit
```

---

## Implementation Patterns

### Pattern 1: Faucet/Sink Validation System

**Complete validation for economic stability**:

```python
class EconomyValidator:
    def __init__(self, economy, player_count):
        self.economy = economy
        self.player_count = player_count

    def validate_stability(self, simulation_days=30):
        '''Simulate economy for N days to check stability'''
        print(f"Simulating economy for {simulation_days} days...")

        total_money = self.calculate_initial_money()
        daily_faucet = self.calculate_daily_faucet()
        daily_sink = self.calculate_daily_sink()

        print(f"\nInitial money supply: {total_money:,}")
        print(f"Daily faucet: {daily_faucet:,}")
        print(f"Daily sink: {daily_sink:,}")
        print(f"Sink/Faucet ratio: {daily_sink/daily_faucet:.2f}")

        # Simulate
        money_over_time = [total_money]
        for day in range(simulation_days):
            total_money += daily_faucet
            total_money -= daily_sink
            money_over_time.append(total_money)

        # Analyze
        final_money = money_over_time[-1]
        money_growth_rate = (final_money / money_over_time[0]) ** (1/simulation_days) - 1

        print(f"\nAfter {simulation_days} days:")
        print(f"  Total money: {final_money:,}")
        print(f"  Growth rate: {money_growth_rate*100:.2f}% per day")

        # Check for inflation
        if money_growth_rate > 0.01:  # >1% growth per day
            print(f"  ⚠️  WARNING: Runaway inflation!")
            print(f"  Money supply growing too fast.")
            print(f"  Increase sinks or reduce faucets.")
            return False

        elif money_growth_rate < -0.01:  # <-1% shrink per day
            print(f"  ⚠️  WARNING: Deflation!")
            print(f"  Money supply shrinking.")
            print(f"  Reduce sinks or increase faucets.")
            return False

        else:
            print(f"  ✓ STABLE: Money supply growth is healthy.")
            return True

    def calculate_initial_money(self):
        '''Money already in economy'''
        return self.player_count * 50000  # Average per player

    def calculate_daily_faucet(self):
        '''Money entering per day'''
        new_players = 100 * 10000               # New player bonuses
        quests = self.player_count * 5 * 5000  # Quests per player
        missions = self.player_count * 10 * 1000  # Missions
        monster_drops = self.player_count * 100 * 50  # Combat

        return new_players + quests + missions + monster_drops

    def calculate_daily_sink(self):
        '''Money leaving per day'''
        transaction_fees = self.player_count * 20 * 1000 * 0.02  # 20 trades/day, 2% fee
        repairs = self.player_count * 5 * 500  # 5 repairs/day
        consumables = self.player_count * 100  # Daily ammo/fuel
        listing_fees = self.player_count * 5 * 1000 * 0.05  # Market listings

        return transaction_fees + repairs + consumables + listing_fees

    def validate_no_arbitrage(self):
        '''Ensure no infinite money loops'''
        print("\nValidating production chains...")

        for recipe in self.economy.recipes:
            input_cost_npc = sum(
                self.economy.npc_sell_price(mat) * qty
                for mat, qty in recipe.inputs.items()
            )

            output_value_npc = self.economy.npc_buy_price(recipe.output)

            profit = output_value_npc - input_cost_npc

            if profit > 0:
                print(f"  ❌ ERROR: {recipe.output} creates infinite money!")
                print(f"     Input cost: {input_cost_npc}")
                print(f"     Output value: {output_value_npc}")
                print(f"     Profit: {profit}")
                return False
            else:
                print(f"  ✓ {recipe.output}: Safe (requires player trading)")

        return True

    def validate_price_bounds(self):
        '''Ensure prices have min/max'''
        print("\nValidating price bounds...")

        for item in self.economy.items:
            if not hasattr(item, 'min_price') or not hasattr(item, 'max_price'):
                print(f"  ❌ ERROR: {item.name} missing price bounds!")
                return False

            if item.min_price <= 0:
                print(f"  ❌ ERROR: {item.name} min_price is {item.min_price} (must be > 0)")
                return False

            if item.max_price < item.min_price * 5:
                print(f"  ⚠️  WARNING: {item.name} max_price too close to min")
                print(f"     Range: {item.min_price} - {item.max_price}")

            print(f"  ✓ {item.name}: {item.min_price} - {item.max_price}")

        return True

    def run_full_validation(self):
        '''Run all validation checks'''
        print("="*60)
        print("ECONOMIC STABILITY VALIDATION")
        print("="*60)

        checks = [
            ("Stability", self.validate_stability),
            ("Arbitrage", self.validate_no_arbitrage),
            ("Price Bounds", self.validate_price_bounds),
        ]

        results = []
        for name, check in checks:
            print(f"\n{name} Check:")
            passed = check()
            results.append((name, passed))

        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        all_passed = all(passed for _, passed in results)

        for name, passed in results:
            status = "✓ PASS" if passed else "❌ FAIL"
            print(f"{status}: {name}")

        if all_passed:
            print("\n✓ Economy is stable and ready for production!")
        else:
            print("\n❌ Economy has critical issues. Fix before launch!")

        return all_passed
```

### Pattern 2: Dynamic Price Adjustment with Stabilization

**Robust price adjustment system**:

```python
class StabilizedPricing:
    def __init__(self, item_name, base_price):
        self.item_name = item_name
        self.base_price = base_price
        self.current_price = base_price

        # Bounds (10% to 10x base)
        self.min_price = base_price * 0.1
        self.max_price = base_price * 10.0

        # Smoothing parameters
        self.ema_alpha = 0.1  # 10% weight to new data
        self.adjustment_rate = 0.01  # 1% change per update

        # History for analysis
        self.price_history = []
        self.trade_history = []

    def update_from_supply_demand(self, supply, demand):
        '''Update price based on supply/demand imbalance'''
        if supply == 0 and demand == 0:
            return  # No activity

        if supply == 0:
            # Pure demand, no supply
            self.current_price *= (1 + self.adjustment_rate)

        elif demand == 0:
            # Pure supply, no demand
            self.current_price *= (1 - self.adjustment_rate)

        else:
            # Calculate ratio
            ratio = demand / supply

            if ratio > 1.1:  # 10% more demand than supply
                # Increase price (dampened)
                self.current_price *= (1 + self.adjustment_rate)

            elif ratio < 0.9:  # 10% more supply than demand
                # Decrease price (dampened)
                self.current_price *= (1 - self.adjustment_rate)

        # Enforce bounds
        self.current_price = clamp(self.current_price, self.min_price, self.max_price)

        # Record
        self.price_history.append(self.current_price)

    def update_from_trades(self, trades):
        '''Update price based on actual trade prices (more accurate)'''
        if len(trades) == 0:
            return

        # Calculate average trade price
        avg_trade_price = sum(t.price * t.quantity for t in trades) / sum(t.quantity for t in trades)

        # Exponential moving average
        self.current_price = (
            self.current_price * (1 - self.ema_alpha) +
            avg_trade_price * self.ema_alpha
        )

        # Enforce bounds
        self.current_price = clamp(self.current_price, self.min_price, self.max_price)

        # Record
        self.price_history.append(self.current_price)
        self.trade_history.extend(trades)

    def get_volatility(self):
        '''Calculate recent price volatility'''
        if len(self.price_history) < 10:
            return 0.0

        recent = self.price_history[-10:]
        mean = sum(recent) / len(recent)
        variance = sum((p - mean)**2 for p in recent) / len(recent)
        std_dev = variance ** 0.5

        return std_dev / mean  # Coefficient of variation

    def should_intervene(self):
        '''Check if NPC intervention is needed'''
        volatility = self.get_volatility()

        # High volatility: Market unstable
        if volatility > 0.2:  # 20% volatility
            return True

        # Price at extremes
        if self.current_price <= self.min_price * 1.1:
            return True  # Near floor
        if self.current_price >= self.max_price * 0.9:
            return True  # Near ceiling

        return False
```

### Pattern 3: NPC Market Maker for Stabilization

**NPCs provide liquidity when markets fail**:

```python
class NPCMarketMaker:
    def __init__(self, item, target_price, liquidity_amount):
        self.item = item
        self.target_price = target_price
        self.liquidity_amount = liquidity_amount  # Max qty NPC will trade

        # Spread (NPC buys low, sells high)
        self.bid_spread = 0.95  # NPC buys at 95% of target
        self.ask_spread = 1.05  # NPC sells at 105% of target

    def get_npc_buy_order(self):
        '''NPC standing offer to buy'''
        price = self.target_price * self.bid_spread
        quantity = self.liquidity_amount

        return {
            'item': self.item,
            'price': price,
            'quantity': quantity,
            'is_npc': True
        }

    def get_npc_sell_order(self):
        '''NPC standing offer to sell'''
        price = self.target_price * self.ask_spread
        quantity = self.liquidity_amount

        return {
            'item': self.item,
            'price': price,
            'quantity': quantity,
            'is_npc': True
        }

    def adjust_target_price(self, current_market_price):
        '''Slowly adjust target towards market price'''
        # Very slow adjustment (1% per update)
        ADJUSTMENT_RATE = 0.01

        self.target_price = (
            self.target_price * (1 - ADJUSTMENT_RATE) +
            current_market_price * ADJUSTMENT_RATE
        )

    def intervene_if_needed(self, market):
        '''Provide liquidity if market is thin'''
        order_book = market.get_order_book(self.item)

        # Check if market has liquidity
        total_buy_orders = sum(o.quantity for o in order_book.buy_orders if not o.is_npc)
        total_sell_orders = sum(o.quantity for o in order_book.sell_orders if not o.is_npc)

        # If market is thin, add NPC orders
        if total_buy_orders < self.liquidity_amount * 0.5:
            market.add_buy_order(self.get_npc_buy_order())

        if total_sell_orders < self.liquidity_amount * 0.5:
            market.add_sell_order(self.get_npc_sell_order())
```

**Effect**:
- Market always has liquidity (players can always trade)
- Prices stabilize near target (NPC absorbs volatility)
- NPC adapts to market (target adjusts slowly)

### Pattern 4: Production Chain Simulator

**Test economic balance before launch**:

```python
class ProductionChainSimulator:
    def __init__(self, economy):
        self.economy = economy

    def simulate_player_behavior(self, num_players=1000, days=30):
        '''Simulate player economy for N days'''
        print(f"Simulating {num_players} players for {days} days...\n")

        # Track metrics
        total_money = num_players * 10000  # Starting credits
        resource_production = {item: 0 for item in self.economy.resources}
        goods_production = {item: 0 for item in self.economy.goods}
        trade_volume = 0

        for day in range(days):
            # Players mine resources
            for resource in self.economy.resources:
                daily_mining = num_players * 100  # 100 units/player/day
                resource_production[resource] += daily_mining

            # Players craft (choose most profitable chain)
            best_recipe = self.find_best_recipe()
            if best_recipe:
                crafters = num_players * 0.3  # 30% of players craft
                daily_crafting = crafters * 2  # 2 items/day
                goods_production[best_recipe.output] += daily_crafting

                # Consume resources
                for mat, qty in best_recipe.inputs.items():
                    resource_production[mat] -= daily_crafting * qty

            # Players trade
            daily_trades = num_players * 10  # 10 trades/player/day
            avg_trade_value = 1000
            trade_volume += daily_trades * avg_trade_value

            # Money faucets
            daily_faucet = self.economy.calculate_daily_faucet(num_players)
            total_money += daily_faucet

            # Money sinks
            daily_sink = self.economy.calculate_daily_sink(num_players)
            total_money -= daily_sink

            # Log every 5 days
            if day % 5 == 0:
                print(f"Day {day}:")
                print(f"  Total money: {total_money:,}")
                print(f"  Trade volume: {trade_volume:,}")
                print(f"  Most produced: {max(goods_production, key=goods_production.get)}")

        # Final report
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE ({days} days)")
        print(f"{'='*60}")

        print(f"\nMoney Supply:")
        print(f"  Initial: {num_players * 10000:,}")
        print(f"  Final: {total_money:,}")
        print(f"  Growth: {(total_money / (num_players * 10000) - 1) * 100:.1f}%")

        print(f"\nResource Production:")
        for resource, qty in resource_production.items():
            print(f"  {resource}: {qty:,}")

        print(f"\nGoods Production:")
        for good, qty in goods_production.items():
            print(f"  {good}: {qty:,}")

        # Check for imbalances
        self.check_for_imbalances(resource_production, goods_production)

    def find_best_recipe(self):
        '''Find most profitable recipe (dominant strategy)'''
        best_recipe = None
        best_profit_per_hour = 0

        for recipe in self.economy.recipes:
            input_cost = sum(
                self.economy.get_price(mat) * qty
                for mat, qty in recipe.inputs.items()
            )
            output_value = self.economy.get_price(recipe.output)
            time_hours = recipe.time / 3600

            profit_per_hour = (output_value - input_cost) / time_hours

            if profit_per_hour > best_profit_per_hour:
                best_profit_per_hour = profit_per_hour
                best_recipe = recipe

        return best_recipe

    def check_for_imbalances(self, resource_production, goods_production):
        '''Detect economic imbalances'''
        print(f"\nImbalance Detection:")

        # Check for unused resources
        for resource, qty in resource_production.items():
            if qty > 100000:  # Excess production
                print(f"  ⚠️  {resource}: Overproduced ({qty:,} units)")
                print(f"     Consider increasing demand or decreasing production.")

            if qty < 0:  # Deficit
                print(f"  ⚠️  {resource}: Deficit ({abs(qty):,} units)")
                print(f"     Production can't keep up with demand.")

        # Check for dominant strategies
        if len(goods_production) > 0:
            max_produced = max(goods_production.values())
            min_produced = min(goods_production.values())

            if max_produced > min_produced * 5:  # 5x imbalance
                print(f"  ⚠️  Production imbalance detected!")
                print(f"     Most produced: {max(goods_production, key=goods_production.get)}")
                print(f"     Least produced: {min(goods_production, key=goods_production.get)}")
                print(f"     Ratio: {max_produced/min_produced:.1f}x")
```

### Pattern 5: Anti-Exploit Validation

**Comprehensive exploit detection**:

```python
class ExploitDetector:
    def __init__(self, economy):
        self.economy = economy

    def find_arbitrage_loops(self):
        '''Detect cycles in production/trading that generate profit'''
        print("Searching for arbitrage loops...\n")

        loops_found = []

        # Check direct buy-sell loops
        for item in self.economy.items:
            npc_sell = self.economy.npc_sell_price(item)
            npc_buy = self.economy.npc_buy_price(item)

            if npc_buy >= npc_sell:
                print(f"❌ CRITICAL: {item} - Direct arbitrage!")
                print(f"   NPC sells for: {npc_sell}")
                print(f"   NPC buys for: {npc_buy}")
                print(f"   Profit: {npc_buy - npc_sell} per unit")
                loops_found.append(('direct', item))

        # Check production loops
        for recipe in self.economy.recipes:
            input_cost = sum(
                self.economy.npc_sell_price(mat) * qty
                for mat, qty in recipe.inputs.items()
            )
            output_value = self.economy.npc_buy_price(recipe.output)

            if output_value > input_cost:
                profit = output_value - input_cost
                print(f"❌ CRITICAL: {recipe.output} - Production arbitrage!")
                print(f"   Input cost: {input_cost}")
                print(f"   Output value: {output_value}")
                print(f"   Profit: {profit}")
                loops_found.append(('production', recipe.output))

        # Check multi-step loops (A→B→C→A)
        # This requires graph traversal to find cycles
        loops_found.extend(self.find_conversion_cycles())

        if len(loops_found) == 0:
            print("✓ No arbitrage loops found!")
        else:
            print(f"\n❌ Found {len(loops_found)} exploitable loops!")

        return loops_found

    def find_conversion_cycles(self):
        '''Find multi-step conversion cycles'''
        # Build conversion graph
        graph = {}
        for recipe in self.economy.recipes:
            # Each recipe is an edge: inputs → output
            graph[recipe.output] = recipe.inputs

        # DFS to find cycles
        cycles = []
        # ... graph traversal logic ...
        return cycles

    def test_market_manipulation(self):
        '''Simulate coordinated buying to check for cornering'''
        print("\nTesting market manipulation resistance...\n")

        for item in self.economy.items:
            # Simulate 100 players buying entire supply
            available_supply = self.economy.get_total_supply(item)
            market_price = self.economy.get_price(item)

            # Cost to buy entire supply
            total_cost = available_supply * market_price

            # Price after buying
            new_price = self.economy.simulate_price_after_buying(item, available_supply)

            price_increase = (new_price / market_price - 1) * 100

            if price_increase > 500:  # 5x price increase
                print(f"⚠️  {item}: Vulnerable to cornering!")
                print(f"   Supply: {available_supply}")
                print(f"   Cost to corner: {total_cost:,}")
                print(f"   Price increase: {price_increase:.0f}%")
                print(f"   Recommendation: Increase supply or add price ceiling")

        print("\n✓ Market manipulation test complete")

    def test_duping_detection(self):
        '''Verify transactions are atomic and validated'''
        print("\nTesting duping prevention...\n")

        # Check if trades are atomic
        if not hasattr(self.economy, 'execute_trade_atomic'):
            print("⚠️  WARNING: No atomic trade execution found!")
            print("   Implement database transactions to prevent duping")

        # Check if inventory is validated
        if not hasattr(self.economy, 'validate_inventory'):
            print("⚠️  WARNING: No inventory validation found!")
            print("   Add checks for negative quantities and overflow")

        print("✓ Duping prevention checks complete")

    def run_all_exploit_tests(self):
        '''Run comprehensive exploit detection'''
        print("="*60)
        print("EXPLOIT DETECTION SUITE")
        print("="*60 + "\n")

        tests = [
            ("Arbitrage Loops", self.find_arbitrage_loops),
            ("Market Manipulation", self.test_market_manipulation),
            ("Duping Prevention", self.test_duping_detection),
        ]

        exploits_found = []

        for name, test in tests:
            print(f"\n{name}:")
            print("-" * 40)
            result = test()
            if result:
                exploits_found.extend(result)

        print("\n" + "="*60)
        print("EXPLOIT DETECTION SUMMARY")
        print("="*60)

        if len(exploits_found) == 0:
            print("✓ No exploits detected!")
        else:
            print(f"❌ Found {len(exploits_found)} potential exploits!")
            print("Fix these before launch!")

        return exploits_found
```

---

## Common Pitfalls

### Pitfall 1: No Money Sinks (Runaway Inflation)

**The Mistake**:
```python
# ❌ Money enters but never leaves
class BrokenEconomy:
    def new_player_joins(self, player):
        player.credits += 10000  # Faucet

    def complete_quest(self, player):
        player.credits += 5000  # Faucet

    # NO SINKS! Money accumulates forever
```

**Why It Fails**:
- Every player brings 10,000 credits
- Quests add 5,000 credits each
- After 1 month: Millions of excess credits
- Hyperinflation: Prices skyrocket
- New players can't afford anything

**Real-World Example**: Early World of Warcraft had insufficient gold sinks. Players accumulated millions of gold, causing runaway inflation. Blizzard added repair costs, mounts, and consumables to drain gold.

**The Fix**:
```python
# ✅ Balance faucets with sinks
class StableEconomy:
    def calculate_faucets(self, player_count):
        daily_faucet = player_count * 10000  # New players + quests
        return daily_faucet

    def calculate_sinks(self, player_count):
        # Transaction fees
        daily_trades = player_count * 10
        trade_fee = daily_trades * 1000 * 0.02

        # Repairs
        daily_repairs = player_count * 5 * 500

        # Consumables (ammo, fuel)
        daily_consumables = player_count * 100

        # Listing fees
        daily_listings = player_count * 5 * 1000 * 0.05

        daily_sink = trade_fee + daily_repairs + daily_consumables + daily_listings
        return daily_sink

    def validate_balance(self, player_count):
        faucet = self.calculate_faucets(player_count)
        sink = self.calculate_sinks(player_count)
        ratio = sink / faucet

        assert 0.8 <= ratio <= 1.2, f"Sink/Faucet ratio is {ratio} (need 0.8-1.2)"
```

**Critical Sinks to Include**:
- Transaction fees (1-5%)
- Repair costs (items degrade)
- Consumables (ammo, fuel, food)
- Listing fees (marketplace)
- Fast travel costs
- Housing/storage fees
- NPC luxury items (cosmetics)

### Pitfall 2: NPC Arbitrage (Infinite Money Loops)

**The Mistake**:
```python
# ❌ NPCs buy and sell at same price
class ExploitableEconomy:
    def npc_sell_price(self, item):
        return self.market_price[item]

    def npc_buy_price(self, item):
        return self.market_price[item]  # SAME!

    # Players discover:
    # Mine 50 Iron (free) → Craft Hull → Sell to NPC
    # Input: 50 Iron @ 10 = 500 credits
    # Output: 1 Hull @ 1000 = 1000 credits
    # Profit: 500 credits (INFINITE LOOP!)
```

**Why It Fails**:
- Players find optimal craft chains
- Everyone grinds the same exploit
- Economy becomes meaningless (everyone prints money)
- Hyperinflation accelerates

**Real-World Example**: Path of Exile had a vendor recipe that generated profit. Players automated it, crashing the economy. GGG nerfed the recipe.

**The Fix**:
```python
# ✅ NPCs buy at discount (bid-ask spread)
class SafeEconomy:
    def npc_sell_price(self, item):
        return self.market_price[item] * 1.0  # 100% (sell to players)

    def npc_buy_price(self, item):
        return self.market_price[item] * 0.6  # 60% (buy from players)

    def validate_no_arbitrage(self):
        for recipe in self.recipes:
            input_cost = sum(
                self.npc_sell_price(mat) * qty
                for mat, qty in recipe.inputs.items()
            )
            output_value = self.npc_buy_price(recipe.output)

            if output_value >= input_cost:
                raise ValueError(f"{recipe.output} creates infinite money!")
```

**Bid-Ask Spread Guidelines**:
- **Safe**: NPCs buy at 50-60% (forces player trading)
- **Moderate**: NPCs buy at 60-70%
- **Risky**: NPCs buy at 80%+ (validate carefully!)

### Pitfall 3: Unbounded Prices (Wild Swings)

**The Mistake**:
```python
# ❌ Prices multiply with no bounds
class UnstablePrices:
    def update_price(self, item, demand, supply):
        if demand > supply:
            self.price[item] *= 1.1  # +10% every update

        # After 50 updates: 10 * (1.1^50) = 1,173 credits!
        # After 100 updates: 137,806 credits (unusable)
```

**Why It Fails**:
- Prices can explode to infinity or crash to zero
- Market becomes unusable
- Players can't plan or budget
- Coordinated manipulation causes chaos

**Real-World Example**: Eve Online had periods of extreme price volatility before implementing market stabilization tools.

**The Fix**:
```python
# ✅ Bounded prices with dampening
class StablePrices:
    def __init__(self, base_price):
        self.price = base_price
        self.min_price = base_price * 0.1   # Floor: 10% of base
        self.max_price = base_price * 10.0  # Ceiling: 10x base

    def update_price(self, demand, supply):
        if demand > supply:
            self.price *= 1.01  # Dampened: +1% not +10%
        else:
            self.price *= 0.99  # Dampened: -1%

        # Enforce bounds
        self.price = clamp(self.price, self.min_price, self.max_price)
```

**Key fixes**:
- Price floors (minimum > 0)
- Price ceilings (maximum < infinity)
- Dampening (1.01x adjustments, not 1.1x)
- Smoothing (EMA, not instant)

### Pitfall 4: Unbalanced Production Chains

**The Mistake**:
```python
# ❌ Some chains way more profitable than others
class ImbalancedChains:
    # Hull: 800 cost → 1000 value = 200 profit (25%)
    # Electronics: 950 cost → 800 value = -150 profit (LOSS!)
    # Fuel Cell: 280 cost → 500 value = 220 profit (79%!)

    # Result: Everyone crafts Fuel Cells, nobody crafts Electronics
```

**Why It Fails**:
- Dominant strategies emerge (one chain is best)
- Other chains become useless
- Market distorts (excess supply of one item)
- Ships can't be built (missing Electronics)

**Real-World Example**: Final Fantasy XIV regularly rebalances crafting recipes to ensure all disciplines are equally profitable.

**The Fix**:
```python
# ✅ Analyze and balance all chains
class BalancedChains:
    def analyze_profit_margins(self):
        for recipe in self.recipes:
            input_cost = sum(self.price[mat] * qty for mat, qty in recipe.inputs.items())
            output_value = self.price[recipe.output]
            time_hours = recipe.time / 3600

            profit_per_hour = (output_value - input_cost) / time_hours

            print(f"{recipe.output}: {profit_per_hour:.0f} credits/hour")

        # Ensure all chains have similar profit/hour (within 2-3x)

    def balance_recipe(self, recipe, target_profit_per_hour):
        current_profit = self.calculate_profit_per_hour(recipe)

        if current_profit < target_profit_per_hour:
            # Increase output value or decrease input cost
            multiplier = target_profit_per_hour / current_profit
            recipe.output_quantity *= multiplier
```

**Balance targets**:
- All chains should be within 2-3x profit/hour
- No chain should be strictly better (dominant strategy)
- Complex chains can pay more (reward knowledge)

### Pitfall 5: No Velocity Control (Resource Flood)

**The Mistake**:
```python
# ❌ Infinite resource production, no consumption
class ResourceFlood:
    def mine(self, player, resource):
        # 100 units/hour per player
        # With 1000 players: 2.4M units/day
        # After 30 days: 72M units (infinite!)
        player.inventory[resource] += 100
```

**Why It Fails**:
- Resources accumulate infinitely
- Supply >> demand always
- Prices crash to near-zero
- Mining becomes pointless (market flooded)

**Real-World Example**: Runescape had resource inflation for years. They added item sinks (high-level equipment degrades and must be repaired with resources).

**The Fix**:
```python
# ✅ Add resource sinks
class ControlledVelocity:
    def consume_fuel(self, player):
        # Travel requires fuel
        player.inventory['Fuel'] -= 10

    def repair_ship(self, player, ship):
        # Repairs consume resources
        player.inventory['Iron'] -= 20

    def decay_food(self, player):
        # Food spoils over time
        for item in player.inventory:
            if item.type == 'food':
                item.durability -= 1
                if item.durability <= 0:
                    player.inventory.remove(item)

    def storage_limits(self, player):
        # Can't hoard infinite resources
        max_storage = 10000
        for resource in player.inventory:
            if player.inventory[resource] > max_storage:
                player.inventory[resource] = max_storage
```

**Resource sinks to add**:
- Consumables (fuel, ammo, food)
- Repairs (items degrade)
- Decay (items expire)
- Storage limits (can't hoard infinitely)
- Crafting failures (chance to lose materials)

### Pitfall 6: New Player Hyperinflation Trap

**The Mistake**:
```python
# ❌ New players join into inflated economy
class NewPlayerTrap:
    def __init__(self):
        # Month 1: Ships cost 3,500
        # Month 6: Ships cost 500,000 (inflation)
        # New player has: 10,000 credits
        # Can't afford basic items!
        pass
```

**Why It Fails**:
- Veterans have millions of credits
- Prices inflate to match veteran wealth
- New players can't afford anything
- New player retention plummets

**Real-World Example**: Eve Online addresses this with "new player areas" where prices are capped and subsidized.

**The Fix**:
```python
# ✅ Scale starting credits with inflation
class NewPlayerProtection:
    def calculate_starting_credits(self):
        # Calculate current price index
        current_price_level = self.calculate_average_price()
        base_price_level = 100  # Launch prices

        inflation_ratio = current_price_level / base_price_level

        # Scale starting credits
        base_starting_credits = 10000
        adjusted_credits = base_starting_credits * inflation_ratio

        return adjusted_credits

    def new_player_marketplace(self, player):
        # Separate market for new players
        if player.account_age_days < 7:
            # Access to subsidized prices
            return self.new_player_market
        else:
            return self.main_market
```

**Protections to add**:
- Scaling starting credits (track inflation)
- New player zones (capped prices)
- Subsidized vendors (sell basics cheaply)
- Progressive taxation (veterans lose money to system)

### Pitfall 7: Market Death Spirals

**The Mistake**:
```python
# ❌ Price spikes cause demand collapse
class DeathSpiral:
    # Platinum spikes to 500 (10x normal)
    # → Nobody buys Platinum
    # → Supply accumulates
    # → Price crashes to 5 (10x too low)
    # → Nobody mines Platinum
    # → Supply dries up
    # → Price spikes again
    # OSCILLATION FOREVER
```

**Why It Fails**:
- Unstable equilibrium (no damping)
- Market never settles
- Players can't rely on prices
- Crafting becomes impossible

**The Fix**:
```python
# ✅ NPC market makers stabilize prices
class StabilizedMarket:
    def __init__(self):
        self.npc_maker = NPCMarketMaker(
            target_price=50,
            liquidity=10000
        )

    def update(self):
        if self.is_market_unstable():
            # NPC provides liquidity to stabilize
            self.npc_maker.provide_liquidity(self.order_book)
```

**Stabilization techniques**:
- NPC market makers (provide liquidity)
- Price floors (NPCs always buy at minimum)
- Supply decay (excess inventory expires)
- Inventory limits (can't stockpile infinitely)

---

## Real-World Examples

### Example 1: Eve Online (Complex Player-Driven Economy)

**Eve Online** has one of the most complex game economies, with:
- 300,000+ active players
- $100+ million USD equivalent traded annually
- Full order book matching
- Regional markets (different prices per system)
- Player corporations (guilds with shared resources)

**Key Patterns**:

```python
# Conceptual Eve economy system
class EveOnlineEconomy:
    def __init__(self):
        # Regional markets (different prices per system)
        self.regions = {}  # {region_id: Market}

        # Full order book matching
        self.order_books = {}  # {(item, region): OrderBook}

        # NPC sinks
        self.npc_seeding = True  # NPCs sell blueprints (faucets)
        self.transaction_tax = 0.025  # 2.5% sales tax (sink)
        self.broker_fee = 0.03  # 3% listing fee (sink)

    def regional_arbitrage(self, item):
        '''Hauling creates gameplay (buy cheap, sell expensive)'''
        # Example: Minerals cheap in mining systems
        jita_price = self.order_books[(item, 'Jita')].get_best_price()
        null_sec_price = self.order_books[(item, 'Null-Sec')].get_best_price()

        profit_per_unit = null_sec_price - jita_price

        # Hauling is risky (pirates, travel time)
        # But profitable if successful
        return profit_per_unit

    def production_chains(self):
        '''Deep production chains (10+ steps)'''
        # Minerals → Components → Subsystems → Ships
        # Each step adds value
        # Complex chains require specialization

        # Example: Building a Titan (supercarrier)
        # - Requires 100+ different materials
        # - Takes 6+ weeks of production time
        # - Costs 70+ billion ISK (700 PLEX ≈ $11,000 USD)

    def destruction_as_sink(self):
        '''Ships are destroyed in combat (major sink)'''
        # When ship explodes, 50% of materials are lost
        # Creates constant demand for new ships
        # Player-driven conflict = economy engine
```

**What Eve Gets Right**:
1. **Strong sinks**: 50% of destroyed ship value removed from game
2. **Regional economies**: Hauling creates gameplay and arbitrage
3. **Deep production chains**: Specialization and interdependence
4. **Player-driven conflict**: PvP creates demand (ships blow up)
5. **Full transparency**: All market data is public (third-party tools)

**Lessons**:
- Destruction is the ultimate sink (items permanently removed)
- Regional markets create hauling gameplay
- Complex chains encourage specialization
- Transparency builds trust

### Example 2: Path of Exile (Currency Item Economy)

**Path of Exile** has no gold. Instead, currency items are:
- Functional (used for crafting)
- Tradeable (player-to-player)
- Self-regulating (supply/demand natural)

**Key Patterns**:

```python
# Conceptual PoE currency system
class PathOfExileEconomy:
    def __init__(self):
        # Currency items are consumable (sinks built-in)
        self.currencies = {
            'Chaos Orb': {'function': 'reroll_item', 'drop_rate': 0.001},
            'Exalted Orb': {'function': 'add_mod', 'drop_rate': 0.0001},
            'Mirror': {'function': 'duplicate_item', 'drop_rate': 0.000001},
        }

        # No NPC trading (player-driven only)
        self.npc_vendors = None  # NPCs only sell basic items

    def currency_as_sink(self, player, item):
        '''Using currency consumes it (automatic sink)'''
        if player.inventory['Chaos Orb'] > 0:
            player.inventory['Chaos Orb'] -= 1  # Consumed!
            item.reroll_mods()  # Item gets random mods

        # This creates natural demand:
        # - Players use currency to craft
        # - Currency is removed from economy
        # - Prices remain stable

    def player_trading(self):
        '''No auction house - player negotiation'''
        # Trade ratios emerge naturally:
        # - 1 Exalted Orb ≈ 150 Chaos Orbs
        # - 1 Mirror ≈ 300 Exalted Orbs ≈ 45,000 Chaos Orbs

        # No NPC prices mean:
        # - Supply/demand sets prices naturally
        # - No arbitrage loops (no NPCs to exploit)
        # - Barter economy (currency for currency)
```

**What PoE Gets Right**:
1. **Currency is consumable**: Using it removes it from economy (built-in sink)
2. **No NPC trading**: Eliminates arbitrage exploits
3. **Functional currency**: Items have inherent value (not abstract credits)
4. **Player-driven prices**: Natural supply/demand equilibrium
5. **Scarcity tiers**: Common to ultra-rare currencies (progression)

**Lessons**:
- Make currency consumable (sinks built-in)
- Eliminate NPCs from core trading (no exploits)
- Functional currency has inherent value
- Let players set prices (emergent economy)

### Example 3: World of Warcraft Auction House

**WoW** has a hybrid economy:
- Player auction house (peer-to-peer)
- NPC vendors (fixed prices)
- Gold faucets (quests, dailies)
- Gold sinks (repairs, mounts, consumables)

**Key Patterns**:

```python
# Conceptual WoW auction house
class WoWAuctionHouse:
    def __init__(self):
        self.listings = []  # Player listings
        self.deposit_fee = 0.05  # 5% to list (sink)
        self.auction_cut = 0.05  # 5% when sold (sink)

    def create_listing(self, seller, item, price, duration):
        '''Player lists item for sale'''
        # Deposit fee (lost even if item doesn't sell)
        deposit = price * self.deposit_fee
        seller.gold -= deposit  # SINK

        listing = {
            'seller': seller,
            'item': item,
            'buyout_price': price,
            'duration': duration
        }
        self.listings.append(listing)

    def buy_listing(self, buyer, listing):
        '''Player buys item'''
        price = listing['buyout_price']

        # Buyer pays full price
        buyer.gold -= price

        # Auction house takes cut (SINK)
        ah_cut = price * self.auction_cut

        # Seller receives (price - cut)
        listing['seller'].gold += (price - ah_cut)

        # Item transferred
        buyer.inventory.add(listing['item'])

    def gold_sinks(self, player):
        '''Various gold sinks'''
        # Repairs (items degrade)
        repair_cost = 100
        player.gold -= repair_cost

        # Mounts (one-time purchase)
        mount_cost = 5000
        player.gold -= mount_cost

        # Consumables (potions, food)
        consumable_cost = 50
        player.gold -= consumable_cost

        # Fast travel
        flight_cost = 10
        player.gold -= flight_cost
```

**What WoW Gets Right**:
1. **Auction house fees**: 5-10% removed from every trade (major sink)
2. **Repair costs**: Items degrade, require gold to fix
3. **One-time purchases**: Mounts, pets, transmog (large sinks)
4. **Consumables**: Constant demand (potions, food, enchants)
5. **NPC luxury items**: Cosmetics, toys (pure sinks)

**Lessons**:
- Transaction fees are effective sinks (every trade removes gold)
- Durability/repairs create constant spending
- One-time purchases (mounts) remove large amounts
- Consumables provide perpetual sinks

### Example 4: Diablo 3 (Failed Economy → Fixed)

**Diablo 3 at launch**:
- Real Money Auction House (RMAH)
- Item drops balanced around trading
- Players could buy best gear (pay-to-win)

**What Went Wrong**:
```python
# Diablo 3 original economy (FAILED)
class Diablo3OriginalEconomy:
    def __init__(self):
        # Real money auction house
        self.rmah = AuctionHouse(currency='USD')

        # Item drops nerfed (force players to trade)
        self.drop_rate_multiplier = 0.1  # 10x lower drops

    def perverse_incentives(self):
        '''Players stop playing, start trading'''
        # Best gear comes from RMAH, not gameplay
        # Players farm gold → buy gear
        # OR: Farm items → sell for $$$

        # Result: Game becomes job, not fun
        # Players quit

# Diablo 3 fixed economy (SUCCESSFUL)
class Diablo3FixedEconomy:
    def __init__(self):
        # RMAH removed entirely
        self.rmah = None

        # No trading (account-bound loot)
        self.trading = False

        # Drop rates massively increased
        self.drop_rate_multiplier = 10.0  # 10x higher

    def loot_as_reward(self):
        '''Playing is rewarding (not trading)'''
        # Best gear comes from playing
        # No economy, no inflation, no exploits
        # Pure game balance
```

**Lessons from Diablo 3**:
1. **Real-money trading is toxic**: Creates pay-to-win, farming bots
2. **Removing economy can work**: Account-bound loot eliminates exploits
3. **Don't force trading**: Let players opt-in to economy
4. **Gameplay should reward players**: Not trading/grinding

### Example 5: Albion Online (Full Loot PvP Economy)

**Albion Online** economy:
- Full loot PvP (killed players drop everything)
- Player-crafted gear (no NPC vendors)
- Localized resources (encourages regional markets)

```python
# Conceptual Albion Online economy
class AlbionOnlineEconomy:
    def __init__(self):
        # All gear is player-crafted
        self.npc_vendors = None

        # Full loot PvP (major sink)
        self.full_loot = True

    def death_as_sink(self, killed_player):
        '''Player death removes items from economy'''
        # Killed player drops ALL gear
        # 50% is destroyed, 50% is loot
        for item in killed_player.equipment:
            if random() < 0.5:
                # Destroyed (SINK)
                item.delete()
            else:
                # Dropped (loot for killer)
                killed_player.position.spawn_loot(item)

        # This creates constant demand for new gear
        # Players must re-equip after death
        # Crafters always have customers

    def localized_resources(self):
        '''Different zones have different resources'''
        # Tier 8 resources only in dangerous zones
        # Forces risk vs reward decisions
        # Creates regional markets (hauling gameplay)
```

**What Albion Gets Right**:
1. **Full loot PvP**: Massive item sink (gear destroyed on death)
2. **Player-crafted economy**: No NPC vendors (pure player-driven)
3. **Localized resources**: Regional markets and hauling
4. **Risk vs reward**: Dangerous zones have best resources

**Lessons**:
- Destruction (death) is powerful sink
- Player-crafting creates interdependence
- Regional resources create hauling gameplay

---

## Cross-References

### Use This Skill WITH:
- **game-balance/economy-balancing**: Overall game economy (XP, rewards, progression)
- **multiplayer-netcode**: Synchronizing economic state across clients
- **database-design**: Storing transactions, inventories, market data
- **anti-cheat**: Preventing duping, botting, and exploits

### Use This Skill AFTER:
- **game-design-fundamentals**: Understanding core loops and player motivations
- **progression-systems**: Balancing rewards with economic constraints
- **systems-thinking**: Understanding feedback loops and equilibrium

### Related Skills:
- **crafting-systems**: Production chains and recipes
- **trading-ui-patterns**: Interface for player marketplaces
- **auction-house-algorithms**: Order matching and price discovery

---

## Testing Checklist

### Pre-Launch Validation
- [ ] **Faucet/Sink Balance**: Sink/Faucet ratio is 0.8-1.2
- [ ] **No Arbitrage**: All production chains are net-negative vs NPCs
- [ ] **Price Bounds**: All items have min/max prices set
- [ ] **Bid-Ask Spread**: NPCs buy at 50-70% of sell price
- [ ] **Production Balance**: All chains within 3x profit/hour of each other
- [ ] **Transaction Fees**: 1-5% fee on all trades (major sink)
- [ ] **Consumables**: Items that require constant spending exist
- [ ] **Repair Costs**: Items degrade and require gold/resources to fix
- [ ] **New Player Protection**: Starting credits scale with inflation OR subsidized market
- [ ] **Rate Limiting**: Daily caps on mining/trading to prevent bots

### Stability Testing
- [ ] **30-Day Simulation**: Simulate 1,000 players for 30 days, verify stable
- [ ] **Exploit Search**: Run exploit detector to find arbitrage loops
- [ ] **Market Manipulation**: Test coordinated buying (cornering)
- [ ] **Bot Resistance**: Verify daily limits prevent 24/7 farming
- [ ] **Death Spiral**: Check if price spikes cause permanent instability

### Post-Launch Monitoring
- [ ] **Track Money Supply**: Log total credits in economy daily
- [ ] **Track Inflation**: Monitor price index over time
- [ ] **Detect Exploits**: Alert if player earns credits too fast
- [ ] **Monitor Imbalances**: Flag if one production chain dominates
- [ ] **New Player Metrics**: Track if new players can afford basics

### Emergency Fixes (If Economy Breaks)
- [ ] **Rollback Database**: Restore to before exploit was discovered
- [ ] **Patch Exploit**: Fix infinite money loop immediately
- [ ] **Emergency Sinks**: Add temporary high-cost NPC items
- [ ] **Ban Exploiters**: Remove profits from players who exploited
- [ ] **Communication**: Announce fixes to player base transparently

---

## Summary

Economic simulation in games requires balancing **faucets** (money entering) and **sinks** (money leaving), validating production chains for **arbitrage exploits**, and implementing **price stabilization** to prevent wild swings. The core principles are:

1. **Faucets < Sinks**: Money entering must be less than money leaving (0.8-1.0 ratio)
2. **Validate chains**: Production chains must be net-negative vs NPCs (no infinite money)
3. **Price bounds**: Set min/max prices (prevent 0 or infinity)
4. **Bid-ask spread**: NPCs buy at 50-70% of sell price (forces player trading)
5. **Balance chains**: All production chains should have similar profit/hour (within 3x)
6. **Resource sinks**: Consumables, repairs, decay (control velocity)
7. **New player protection**: Scale starting credits with inflation OR subsidized markets
8. **Test before launch**: Run 30-day simulation, exploit detection, balance analysis

Master these patterns and avoid the common pitfalls (no sinks, NPC arbitrage, unbounded prices, unbalanced chains), and your economy will be stable, engaging, and exploit-resistant for months or years.
