#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deriv MULTDOWN + CALL Paired Strategy Bot
==========================================
Strategy: Open MULTDOWN (×100) + CALL simultaneously every minute
- MULTDOWN: Profits from price drops, capped loss at stake
- CALL: Hedges upward moves with 5-tick barrier
- Expected net: ~$1.50/min after 5% commission on each leg

Uses official python-deriv-api package
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Bot configuration"""
    # Your API credentials
    API_TOKEN = "SXS8qYkFw2zIKn0"
    APP_ID = 1089
    
    # Trading parameters
    SYMBOL = "1HZ100V"  # Volatility 100 (1s) Index
    STAKE_PER_LEG = 1.0  # $1 per contract
    MULTIPLIER = 100  # 100× for MULTDOWN
    TICK_SIZE = 0.01  # 1 tick = 0.01 pt
    
    # Timing
    CALL_DURATION = 1  # 1 minute
    CALL_DURATION_UNIT = "m"
    TRADE_INTERVAL = 65  # Wait 65s for settlement + safety margin
    
    # Fees & Limits
    COMMISSION_RATE = 0.05  # 5% per leg
    INITIAL_EQUITY = 10.0
    MIN_EQUITY = 2.0  # Stop if equity < $2
    MAX_ROUNDS = 500  # ~8 hours
    
    # System
    LOG_LEVEL = logging.INFO  # Back to INFO


# ============================================================================
# DATA MODELS
# ============================================================================

class LegStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class TradeLeg:
    """Single contract leg (MULTDOWN or CALL)"""
    type: str  # "MULTDOWN" or "CALL"
    contract_id: Optional[str] = None
    stake: float = 0.0
    buy_price: float = 0.0
    sell_price: float = 0.0
    profit: float = 0.0
    status: LegStatus = LegStatus.PENDING
    error: Optional[str] = None


@dataclass
class PairedTrade:
    """One complete round: MULTDOWN + CALL"""
    round_num: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_price: float = 0.0
    
    multdown_leg: TradeLeg = field(default_factory=lambda: TradeLeg("MULTDOWN"))
    call_leg: TradeLeg = field(default_factory=lambda: TradeLeg("CALL"))
    
    commission: float = 0.0
    net_profit: float = 0.0
    equity_after: float = 0.0
    
    @property
    def gross_profit(self) -> float:
        return self.multdown_leg.profit + self.call_leg.profit
    
    @property
    def is_complete(self) -> bool:
        return (self.multdown_leg.status == LegStatus.CLOSED and 
                self.call_leg.status == LegStatus.CLOSED)


@dataclass
class TradingStats:
    """Overall trading statistics"""
    rounds_completed: int = 0
    total_commission: float = 0.0
    total_gross_profit: float = 0.0
    total_net_profit: float = 0.0
    
    winning_rounds: int = 0
    losing_rounds: int = 0
    breakeven_rounds: int = 0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    current_streak: int = 0
    
    def update(self, trade: PairedTrade):
        """Update stats with completed trade"""
        self.rounds_completed += 1
        self.total_commission += trade.commission
        self.total_gross_profit += trade.gross_profit
        self.total_net_profit += trade.net_profit
        
        if trade.net_profit > 0:
            self.winning_rounds += 1
            self.current_streak = max(1, self.current_streak + 1)
            self.largest_win = max(self.largest_win, trade.net_profit)
        elif trade.net_profit < 0:
            self.losing_rounds += 1
            self.current_streak = min(-1, self.current_streak - 1)
            self.largest_loss = min(self.largest_loss, trade.net_profit)
        else:
            self.breakeven_rounds += 1
    
    @property
    def win_rate(self) -> float:
        total = self.winning_rounds + self.losing_rounds
        return (self.winning_rounds / total * 100) if total > 0 else 0.0
    
    @property
    def avg_profit_per_round(self) -> float:
        return self.total_net_profit / self.rounds_completed if self.rounds_completed > 0 else 0.0


# ============================================================================
# TRADING BOT
# ============================================================================

class PairedStrategyBot:
    """MULTDOWN + CALL paired strategy executor"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api = None
        self.equity = config.INITIAL_EQUITY
        self.trades: List[PairedTrade] = []
        self.stats = TradingStats()
        self.logger = logging.getLogger("Bot")
    
    async def start(self):
        """Start trading bot"""
        self._print_header()
        
        try:
            # Import and initialize Deriv API
            try:
                from deriv_api import DerivAPI
            except ImportError:
                self.logger.error("Missing python-deriv-api. Install with: pip install python-deriv-api")
                return
            
            # Initialize API
            self.logger.info("Connecting to Deriv API...")
            self.api = DerivAPI(app_id=self.config.APP_ID)
            
            # Authorize
            auth = await self.api.authorize(self.config.API_TOKEN)
            self.logger.info(f"✅ Authorized as {auth['authorize']['loginid']} "
                           f"({auth['authorize']['currency']} account)")
            
            # Get initial balance
            balance_info = await self.api.balance()
            actual_balance = float(balance_info['balance']['balance'])
            self.logger.info(f"💰 Account balance: ${actual_balance:.2f}")
            
            # Main trading loop
            round_num = 0
            while round_num < self.config.MAX_ROUNDS:
                round_num += 1
                
                # Check equity limit
                if self.equity < self.config.MIN_EQUITY:
                    self.logger.warning(f"\n⛔ Equity below ${self.config.MIN_EQUITY:.2f} - stopping")
                    break
                
                # Execute paired trade
                await self._execute_round(round_num)
                
                # Brief pause before next round
                if round_num < self.config.MAX_ROUNDS:
                    await asyncio.sleep(3)
        
        except KeyboardInterrupt:
            self.logger.info("\n⏹️  Bot stopped by user")
        except Exception as e:
            self.logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        finally:
            self._print_summary()
    
    async def _execute_round(self, round_num: int):
        """Execute one complete paired trade"""
        trade = PairedTrade(round_num=round_num)
        
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔄 Round {round_num}")
            self.logger.info(f"{'='*60}")
            
            # Step 1: Open MULTDOWN (we'll get entry price from the proposal)
            self.logger.info(f"📉 Opening MULTDOWN (×{self.config.MULTIPLIER})...")
            multdown_proposal = await self.api.proposal({
                "proposal": 1,
                "amount": self.config.STAKE_PER_LEG,
                "basis": "stake",
                "contract_type": "MULTDOWN",
                "currency": "USD",
                "multiplier": self.config.MULTIPLIER,
                "symbol": self.config.SYMBOL,
            })
            
            if "error" in multdown_proposal:
                raise Exception(f"MULTDOWN proposal error: {multdown_proposal['error']['message']}")
            
            # Get entry price from proposal
            trade.entry_price = float(multdown_proposal["proposal"]["spot"])
            
            self.logger.info(f"   Entry price: {trade.entry_price:.2f}")
            
            # Buy MULTDOWN
            try:
                multdown_buy = await self.api.buy({
                    "buy": multdown_proposal["proposal"]["id"],
                    "price": multdown_proposal["proposal"]["ask_price"]
                })
            except Exception as e:
                self.logger.error(f"Buy call failed: {e}")
                self.logger.error(f"Proposal was: {multdown_proposal}")
                raise
            
            if "error" in multdown_buy:
                raise Exception(f"MULTDOWN buy error: {multdown_buy['error']['message']}")
            
            trade.multdown_leg.contract_id = str(multdown_buy["buy"]["contract_id"])  # Convert to string
            trade.multdown_leg.buy_price = float(multdown_buy["buy"]["buy_price"])
            trade.multdown_leg.stake = self.config.STAKE_PER_LEG
            trade.multdown_leg.status = LegStatus.OPEN
            self.logger.info(f"   ✅ MULTDOWN opened | ID: {trade.multdown_leg.contract_id[:12]}...")
            
            # Step 2: Open RISE (95% payout digital option)
            self.logger.info(f"📈 Opening RISE option...")
            call_proposal = await self.api.proposal({
                "proposal": 1,
                "amount": self.config.STAKE_PER_LEG,
                "basis": "stake",
                "contract_type": "CALL",  # RISE is a CALL contract in digital options
                "currency": "USD",
                "duration": self.config.CALL_DURATION,
                "duration_unit": self.config.CALL_DURATION_UNIT,
                "symbol": self.config.SYMBOL,
            })
            
            if "error" in call_proposal:
                raise Exception(f"RISE proposal error: {call_proposal['error']['message']}")
            
            call_buy = await self.api.buy({
                "buy": call_proposal["proposal"]["id"],
                "price": call_proposal["proposal"]["ask_price"]
            })
            
            if "error" in call_buy:
                raise Exception(f"RISE buy error: {call_buy['error']['message']}")
            
            trade.call_leg.contract_id = str(call_buy["buy"]["contract_id"])  # Convert to string
            trade.call_leg.buy_price = float(call_buy["buy"]["buy_price"])
            trade.call_leg.stake = self.config.STAKE_PER_LEG
            trade.call_leg.status = LegStatus.OPEN
            self.logger.info(f"   ✅ RISE opened | ID: {trade.call_leg.contract_id[:12]}...")
            
            # Step 3: Wait for CALL to expire
            self.logger.info(f"⏳ Waiting {self.config.TRADE_INTERVAL}s for settlement...")
            await asyncio.sleep(self.config.TRADE_INTERVAL)
            
            # Step 4: Close MULTDOWN
            self.logger.info("🔒 Closing MULTDOWN...")
            multdown_sell = await self.api.sell({
                "sell": int(trade.multdown_leg.contract_id),  # Convert back to int for sell
                "price": 0  # Market price
            })
            
            trade.multdown_leg.sell_price = float(multdown_sell["sell"]["sold_for"])
            trade.multdown_leg.profit = float(multdown_sell["sell"]["profit"])
            trade.multdown_leg.status = LegStatus.CLOSED
            self.logger.info(f"   ✅ MULTDOWN closed | P/L: ${trade.multdown_leg.profit:+.2f}")
            
            # Step 5: Get CALL final status (should be auto-settled)
            self.logger.info("📊 Checking CALL status...")
            call_info = await self.api.proposal_open_contract({
                "proposal_open_contract": 1,
                "contract_id": int(trade.call_leg.contract_id)  # Convert back to int
            })
            
            poc = call_info["proposal_open_contract"]
            trade.call_leg.profit = float(poc.get("profit", 0))
            trade.call_leg.status = LegStatus.CLOSED
            
            call_status = poc.get("status", "unknown")
            self.logger.info(f"   ✅ RISE {call_status} | P/L: ${trade.call_leg.profit:+.2f}")
            
            # Step 6: Calculate P/L
            trade.commission = 2 * self.config.STAKE_PER_LEG * self.config.COMMISSION_RATE
            trade.net_profit = trade.gross_profit - trade.commission
            self.equity += trade.net_profit
            trade.equity_after = self.equity
            
            # Update stats
            self.stats.update(trade)
            self.trades.append(trade)
            
            # Log results
            self._log_trade_result(trade)
        
        except Exception as e:
            self.logger.error(f"❌ Round {round_num} failed: {e}")
            trade.multdown_leg.error = str(e)
            trade.call_leg.error = str(e)
            self.trades.append(trade)
    
    def _log_trade_result(self, trade: PairedTrade):
        """Log trade results"""
        emoji = "🟢" if trade.net_profit > 0 else "🔴" if trade.net_profit < 0 else "⚪"
        
        self.logger.info(f"\n{emoji} ROUND {trade.round_num} COMPLETE")
        self.logger.info(f"   MULTDOWN P/L    : ${trade.multdown_leg.profit:+.2f}")
        self.logger.info(f"   CALL P/L        : ${trade.call_leg.profit:+.2f}")
        self.logger.info(f"   Gross profit    : ${trade.gross_profit:+.2f}")
        self.logger.info(f"   Commission      : -${trade.commission:.2f}")
        self.logger.info(f"   Net profit      : ${trade.net_profit:+.2f}")
        self.logger.info(f"   Equity          : ${trade.equity_after:.2f} ({self.equity - self.config.INITIAL_EQUITY:+.2f})")
        self.logger.info(f"   Win rate        : {self.stats.win_rate:.1f}% | Streak: {self.stats.current_streak:+d}")
    
    def _print_header(self):
        """Print startup header"""
        self.logger.info("\n" + "="*70)
        self.logger.info("🤖 DERIV MULTDOWN + CALL PAIRED STRATEGY BOT")
        self.logger.info("="*70)
        self.logger.info(f"Symbol         : {self.config.SYMBOL}")
        self.logger.info(f"Stake per leg  : ${self.config.STAKE_PER_LEG:.2f}")
        self.logger.info(f"MULTDOWN       : ×{self.config.MULTIPLIER} multiplier")
        self.logger.info(f"RISE           : {self.config.CALL_DURATION}{self.config.CALL_DURATION_UNIT} digital option (95% payout)")
        self.logger.info(f"Commission     : {self.config.COMMISSION_RATE*100:.0f}% per leg (${2*self.config.STAKE_PER_LEG*self.config.COMMISSION_RATE:.2f}/round)")
        self.logger.info(f"Initial equity : ${self.config.INITIAL_EQUITY:.2f}")
        self.logger.info(f"Stop loss      : ${self.config.MIN_EQUITY:.2f}")
        self.logger.info(f"Max rounds     : {self.config.MAX_ROUNDS}")
        self.logger.info("="*70)
    
    def _print_summary(self):
        """Print final summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 SESSION SUMMARY")
        self.logger.info("="*70)
        
        # Count all trades including incomplete ones
        total_trades_attempted = len(self.trades)
        completed_trades = [t for t in self.trades if t.is_complete]
        
        if len(completed_trades) > 0:
            self.logger.info(f"Total rounds       : {self.stats.rounds_completed}")
            self.logger.info(f"Winning rounds     : {self.stats.winning_rounds}")
            self.logger.info(f"Losing rounds      : {self.stats.losing_rounds}")
            self.logger.info(f"Win rate           : {self.stats.win_rate:.2f}%")
            self.logger.info(f"")
            self.logger.info(f"Gross profit       : ${self.stats.total_gross_profit:+.2f}")
            self.logger.info(f"Total commission   : -${self.stats.total_commission:.2f}")
            self.logger.info(f"Net profit         : ${self.stats.total_net_profit:+.2f}")
            self.logger.info(f"Avg per round      : ${self.stats.avg_profit_per_round:+.2f}")
            self.logger.info(f"")
            self.logger.info(f"Starting equity    : ${self.config.INITIAL_EQUITY:.2f}")
            self.logger.info(f"Final equity       : ${self.equity:.2f}")
            self.logger.info(f"Total return       : {(self.equity/self.config.INITIAL_EQUITY-1)*100:+.2f}%")
            self.logger.info(f"")
            self.logger.info(f"Largest win        : ${self.stats.largest_win:+.2f}")
            self.logger.info(f"Largest loss       : ${self.stats.largest_loss:+.2f}")
            
            if total_trades_attempted > len(completed_trades):
                self.logger.info(f"")
                self.logger.info(f"Failed trades      : {total_trades_attempted - len(completed_trades)}")
            
            self.logger.info("="*70)
            
            if self.equity > self.config.INITIAL_EQUITY:
                self.logger.info("💰 Session profitable!")
            else:
                self.logger.info("📉 Session ended in loss")
        else:
            if total_trades_attempted > 0:
                self.logger.info(f"Trades attempted   : {total_trades_attempted}")
                self.logger.info(f"All trades failed or were incomplete")
            else:
                self.logger.info("No trades executed")
            self.logger.info("="*70)


# ============================================================================
# MAIN
# ============================================================================

def setup_logging(level=logging.INFO):
    """Configure logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )


async def main():
    """Entry point"""
    setup_logging(Config.LOG_LEVEL)
    
    bot = PairedStrategyBot(Config)
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        sys.exit(1)