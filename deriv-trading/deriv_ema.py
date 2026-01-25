#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deriv EMA Crossover Trading Bot
================================
Strategy: 100-period EMA crossover on 1-minute timeframe
- Long: When candle closes above EMA(100)
- Short: When candle closes below EMA(100)
- Trades remain open until opposite signal occurs (no time-based expiry)
- Only one position open at a time
- Configurable stake amount (default: $2000)

Uses official python-deriv-api package
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

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
    STAKE_AMOUNT = 2000.0  # Configurable stake amount
    MULTIPLIER = 100  # 100× multiplier
    
    # EMA parameters
    EMA_LENGTH = 100  # EMA period length
    EMA_OFFSET = 0  # EMA offset
    
    # Timeframe - used only for candlestick analysis and EMA calculation
    TIMEFRAME = "1m"  # 1-minute timeframe
    CANDLE_COUNT = 150  # Number of candles to fetch (more than EMA length)
    
    # Timing
    SIGNAL_CHECK_INTERVAL = 60  # Check for new signals every 60 seconds
    WAIT_AFTER_CLOSE = 5  # Wait 5 seconds after closing a position before opening new one
    
    # Fees & Limits
    COMMISSION_RATE = 0.0  # No commission for multiplier contracts
    INITIAL_EQUITY = 10000.0
    MIN_EQUITY = 2000.0  # Stop if equity < $2000
    MAX_ROUNDS = 1000  # Maximum number of signal checks
    
    # System
    LOG_LEVEL = logging.INFO

# ============================================================================
# DATA MODELS
# ============================================================================

class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    ERROR = "error"

@dataclass
class Position:
    """Single trading position - remains open until opposite signal"""
    position_type: PositionType
    contract_id: Optional[str] = None
    stake: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    profit: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class CandleData:
    """Candlestick data for analysis"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

@dataclass
class EMAStrategyBot:
    """EMA Crossover Trading Bot - positions follow the trend"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api = None
        self.equity = config.INITIAL_EQUITY
        self.current_position: Optional[Position] = None
        self.candle_history: List[CandleData] = []
        self.ema_values: List[float] = []
        self.trades: List[Position] = []
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'current_streak': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        self.logger = logging.getLogger("EMABot")
    
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
            
            # Main trading loop - checks for signals every interval
            round_num = 0
            while round_num < self.config.MAX_ROUNDS:
                round_num += 1
                
                # Check equity limit
                if self.equity < self.config.MIN_EQUITY:
                    self.logger.warning(f"\n⛔ Equity below ${self.config.MIN_EQUITY:.2f} - stopping")
                    break
                
                # Execute trading round (check signals)
                await self._execute_round(round_num)
                
                # Wait for next signal check
                if round_num < self.config.MAX_ROUNDS:
                    await asyncio.sleep(self.config.SIGNAL_CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            self.logger.info("\n⏹️  Bot stopped by user")
        except Exception as e:
            self.logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        finally:
            self._print_summary()
    
    async def _execute_round(self, round_num: int):
        """Execute one trading round - check for signals and manage positions"""
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔄 Signal Check {round_num}")
            self.logger.info(f"{'='*60}")
            
            # Step 1: Get latest candlestick data
            await self._get_candle_data()
            
            if len(self.candle_history) < self.config.EMA_LENGTH:
                self.logger.warning(f"⚠️  Not enough data for EMA calculation. Need {self.config.EMA_LENGTH} candles, got {len(self.candle_history)}")
                return
            
            # Step 2: Calculate EMA
            await self._calculate_ema()
            
            # Step 3: Get current candle and EMA values
            current_candle = self.candle_history[-1]
            current_ema = self.ema_values[-1]
            
            self.logger.info(f"📊 Current Price: {current_candle.close:.2f} | EMA({self.config.EMA_LENGTH}): {current_ema:.2f}")
            
            # Step 4: Check for trading signals and manage positions
            await self._check_trading_signals(current_candle, current_ema)
            
        except Exception as e:
            self.logger.error(f"❌ Signal check {round_num} failed: {e}")
    
    async def _get_candle_data(self):
        """Get candlestick data from Deriv API for analysis"""
        try:
            # Calculate end time (now) and start time
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=self.config.CANDLE_COUNT)
            
            # Get candle data
            candles_response = await self.api.ticks_history({
                "ticks_history": self.config.SYMBOL,
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "style": "candles",
                "granularity": 60  # 1-minute candles for analysis
            })
            
            if "error" in candles_response:
                raise Exception(f"Error getting candle data: {candles_response['error']['message']}")
            
            # Parse candle data
            self.candle_history = []
            for candle in candles_response["candles"]:
                candle_data = CandleData(
                    timestamp=datetime.fromtimestamp(candle["epoch"], timezone.utc),
                    open=float(candle["open"]),
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    close=float(candle["close"])
                )
                self.candle_history.append(candle_data)
            
            self.logger.info(f"📈 Retrieved {len(self.candle_history)} candlesticks for analysis")
            
        except Exception as e:
            self.logger.error(f"❌ Error getting candle data: {e}")
            raise
    
    async def _calculate_ema(self):
        """Calculate Exponential Moving Average for trend analysis"""
        try:
            # Extract closing prices
            close_prices = [candle.close for candle in self.candle_history]
            
            # Calculate EMA using pandas
            df = pd.DataFrame({'close': close_prices})
            df['EMA'] = df['close'].ewm(span=self.config.EMA_LENGTH, adjust=False).mean()
            
            # Store EMA values
            self.ema_values = df['EMA'].tolist()
            
            self.logger.debug(f"📊 Calculated EMA({self.config.EMA_LENGTH}) for {len(self.ema_values)} periods")
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating EMA: {e}")
            raise
    
    async def _check_trading_signals(self, current_candle: CandleData, current_ema: float):
        """
        Check for trading signals based on EMA crossover
        Positions remain open until opposite signal occurs
        """
        try:
            # Previous candle and EMA values for crossover detection
            if len(self.candle_history) < 2 or len(self.ema_values) < 2:
                return
            
            prev_candle = self.candle_history[-2]
            prev_ema = self.ema_values[-2]
            
            # Current position status
            has_open_position = self.current_position is not None and self.current_position.status == PositionStatus.OPEN
            
            # Signal conditions - detect crossovers
            price_above_ema = current_candle.close > current_ema
            price_below_ema = current_candle.close < current_ema
            prev_price_above_ema = prev_candle.close > prev_ema
            prev_price_below_ema = prev_candle.close < prev_ema
            
            # Long signal: price crosses above EMA
            long_signal = price_above_ema and prev_price_below_ema
            
            # Short signal: price crosses below EMA
            short_signal = price_below_ema and prev_price_above_ema
            
            self.logger.debug(f"📊 Signals - Long: {long_signal}, Short: {short_signal}")
            
            # Execute trades based on signals
            if long_signal and not has_open_position:
                await self._open_long_position(current_candle.close)
            elif short_signal and not has_open_position:
                await self._open_short_position(current_candle.close)
            elif has_open_position:
                # Check if we should close the position based on opposite signal
                await self._check_position_closure(current_candle.close, current_ema)
            
        except Exception as e:
            self.logger.error(f"❌ Error checking trading signals: {e}")
    
    async def _open_long_position(self, entry_price: float):
        """Open a long position - remains open until short signal"""
        try:
            self.logger.info(f"📈 Opening LONG position at {entry_price:.2f}")
            
            # Create contract proposal
            proposal = await self.api.proposal({
                "proposal": 1,
                "amount": self.config.STAKE_AMOUNT,
                "basis": "stake",
                "contract_type": "MULTIUP",  # Long position
                "currency": "USD",
                "multiplier": self.config.MULTIPLIER,
                "symbol": self.config.SYMBOL,
            })
            
            if "error" in proposal:
                raise Exception(f"Proposal error: {proposal['error']['message']}")
            
            # Buy the contract
            buy_response = await self.api.buy({
                "buy": proposal["proposal"]["id"],
                "price": proposal["proposal"]["ask_price"]
            })
            
            if "error" in buy_response:
                raise Exception(f"Buy error: {buy_response['error']['message']}")
            
            # Create position object
            self.current_position = Position(
                position_type=PositionType.LONG,
                contract_id=str(buy_response["buy"]["contract_id"]),
                stake=self.config.STAKE_AMOUNT,
                entry_price=entry_price,
                open_time=datetime.now(timezone.utc),
                status=PositionStatus.OPEN
            )
            
            self.logger.info(f"✅ LONG position opened | ID: {self.current_position.contract_id[:12]}... | Stake: ${self.config.STAKE_AMOUNT:.2f}")
            self.logger.info(f"📊 Position will remain open until price crosses below EMA")
            
        except Exception as e:
            self.logger.error(f"❌ Error opening long position: {e}")
    
    async def _open_short_position(self, entry_price: float):
        """Open a short position - remains open until long signal"""
        try:
            self.logger.info(f"📉 Opening SHORT position at {entry_price:.2f}")
            
            # Create contract proposal
            proposal = await self.api.proposal({
                "proposal": 1,
                "amount": self.config.STAKE_AMOUNT,
                "basis": "stake",
                "contract_type": "MULTIDOWN",  # Short position
                "currency": "USD",
                "multiplier": self.config.MULTIPLIER,
                "symbol": self.config.SYMBOL,
            })
            
            if "error" in proposal:
                raise Exception(f"Proposal error: {proposal['error']['message']}")
            
            # Buy the contract
            buy_response = await self.api.buy({
                "buy": proposal["proposal"]["id"],
                "price": proposal["proposal"]["ask_price"]
            })
            
            if "error" in buy_response:
                raise Exception(f"Buy error: {buy_response['error']['message']}")
            
            # Create position object
            self.current_position = Position(
                position_type=PositionType.SHORT,
                contract_id=str(buy_response["buy"]["contract_id"]),
                stake=self.config.STAKE_AMOUNT,
                entry_price=entry_price,
                open_time=datetime.now(timezone.utc),
                status=PositionStatus.OPEN
            )
            
            self.logger.info(f"✅ SHORT position opened | ID: {self.current_position.contract_id[:12]}... | Stake: ${self.config.STAKE_AMOUNT:.2f}")
            self.logger.info(f"📊 Position will remain open until price crosses above EMA")
            
        except Exception as e:
            self.logger.error(f"❌ Error opening short position: {e}")
    
    async def _check_position_closure(self, current_price: float, current_ema: float):
        """
        Check if current position should be closed based on opposite signal
        This is the only way positions are closed - no time-based expiry
        """
        try:
            if not self.current_position or self.current_position.status != PositionStatus.OPEN:
                return
            
            # Close conditions based on opposite signal
            should_close = False
            
            if self.current_position.position_type == PositionType.LONG:
                # Close long if price crosses below EMA (short signal)
                should_close = current_price < current_ema
            elif self.current_position.position_type == PositionType.SHORT:
                # Close short if price crosses above EMA (long signal)
                should_close = current_price > current_ema
            
            if should_close:
                await self._close_position(current_price)
            
        except Exception as e:
            self.logger.error(f"❌ Error checking position closure: {e}")
    
    async def _close_position(self, exit_price: float):
        """Close the current position due to opposite trading signal"""
        try:
            if not self.current_position:
                return
            
            position_type = self.current_position.position_type.value.upper()
            self.logger.info(f"🔒 Closing {position_type} position at {exit_price:.2f} (opposite signal detected)")
            
            # Sell the contract
            sell_response = await self.api.sell({
                "sell": int(self.current_position.contract_id),
                "price": 0  # Market price
            })
            
            if "error" in sell_response:
                raise Exception(f"Sell error: {sell_response['error']['message']}")
            
            # Update position
            self.current_position.exit_price = exit_price
            self.current_position.close_time = datetime.now(timezone.utc)
            self.current_position.profit = float(sell_response["sell"]["profit"])
            self.current_position.status = PositionStatus.CLOSED
            
            # Calculate position duration
            duration = self.current_position.close_time - self.current_position.open_time
            
            # Update equity and stats
            self.equity += self.current_position.profit
            self._update_stats(self.current_position)
            
            # Log results
            self._log_trade_result(self.current_position, duration)
            
            # Add to trades list
            self.trades.append(self.current_position)
            
            # Clear current position
            self.current_position = None
            
            # Wait before potentially opening new position
            await asyncio.sleep(self.config.WAIT_AFTER_CLOSE)
            
        except Exception as e:
            self.logger.error(f"❌ Error closing position: {e}")
    
    def _update_stats(self, position: Position):
        """Update trading statistics"""
        self.stats['total_trades'] += 1
        
        if position.profit > 0:
            self.stats['winning_trades'] += 1
            self.stats['current_streak'] = max(1, self.stats['current_streak'] + 1)
            self.stats['largest_win'] = max(self.stats['largest_win'], position.profit)
        elif position.profit < 0:
            self.stats['losing_trades'] += 1
            self.stats['current_streak'] = min(-1, self.stats['current_streak'] - 1)
            self.stats['largest_loss'] = min(self.stats['largest_loss'], position.profit)
        
        self.stats['total_profit'] += position.profit
    
    def _log_trade_result(self, position: Position, duration: timedelta):
        """Log trade results with duration"""
        emoji = "🟢" if position.profit > 0 else "🔴" if position.profit < 0 else "⚪"
        
        # Format duration
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        self.logger.info(f"\n{emoji} TRADE CLOSED")
        self.logger.info(f"   Type            : {position.position_type.value.upper()}")
        self.logger.info(f"   Entry Price     : {position.entry_price:.2f}")
        self.logger.info(f"   Exit Price      : {position.exit_price:.2f}")
        self.logger.info(f"   Duration        : {duration_str}")
        self.logger.info(f"   Profit/Loss     : ${position.profit:+.2f}")
        self.logger.info(f"   Equity          : ${self.equity:.2f}")
        self.logger.info(f"   Win Rate        : {self._get_win_rate():.1f}%")
        self.logger.info(f"   Current Streak  : {self.stats['current_streak']:+d}")
    
    def _get_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.stats['total_trades'] == 0:
            return 0.0
        return (self.stats['winning_trades'] / self.stats['total_trades']) * 100
    
    def _print_header(self):
        """Print startup header"""
        self.logger.info("\n" + "="*70)
        self.logger.info("🤖 DERIV EMA CROSSOVER TRADING BOT")
        self.logger.info("="*70)
        self.logger.info(f"Symbol         : {self.config.SYMBOL}")
        self.logger.info(f"Analysis TF     : {self.config.TIMEFRAME} (for EMA calculation only)")
        self.logger.info(f"EMA Period     : {self.config.EMA_LENGTH}")
        self.logger.info(f"Stake Amount   : ${self.config.STAKE_AMOUNT:.2f}")
        self.logger.info(f"Multiplier     : ×{self.config.MULTIPLIER}")
        self.logger.info(f"Position Style : Trend-following (no time expiry)")
        self.logger.info(f"Initial Equity : ${self.config.INITIAL_EQUITY:.2f}")
        self.logger.info(f"Stop Loss      : ${self.config.MIN_EQUITY:.2f}")
        self.logger.info(f"Max Signal Checks : {self.config.MAX_ROUNDS}")
        self.logger.info("="*70)
    
    def _print_summary(self):
        """Print final summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 SESSION SUMMARY")
        self.logger.info("="*70)
        
        if self.stats['total_trades'] > 0:
            self.logger.info(f"Total Trades     : {self.stats['total_trades']}")
            self.logger.info(f"Winning Trades   : {self.stats['winning_trades']}")
            self.logger.info(f"Losing Trades    : {self.stats['losing_trades']}")
            self.logger.info(f"Win Rate         : {self._get_win_rate():.2f}%")
            self.logger.info(f"")
            self.logger.info(f"Total Profit     : ${self.stats['total_profit']:+.2f}")
            self.logger.info(f"")
            self.logger.info(f"Starting Equity  : ${self.config.INITIAL_EQUITY:.2f}")
            self.logger.info(f"Final Equity     : ${self.equity:.2f}")
            self.logger.info(f"Total Return     : {(self.equity/self.config.INITIAL_EQUITY-1)*100:+.2f}%")
            self.logger.info(f"")
            self.logger.info(f"Largest Win      : ${self.stats['largest_win']:+.2f}")
            self.logger.info(f"Largest Loss     : ${self.stats['largest_loss']:+.2f}")
            
            # Calculate average trade duration if we have closed positions
            if self.trades:
                total_duration = sum([(trade.close_time - trade.open_time).total_seconds() 
                                    for trade in self.trades if trade.close_time], 0)
                avg_duration_seconds = total_duration / len(self.trades)
                avg_duration = timedelta(seconds=avg_duration_seconds)
                self.logger.info(f"Avg Trade Duration : {str(avg_duration).split('.')[0]}")
            
            self.logger.info("="*70)
            
            if self.equity > self.config.INITIAL_EQUITY:
                self.logger.info("💰 Session profitable!")
            else:
                self.logger.info("📉 Session ended in loss")
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
    
    bot = EMAStrategyBot(Config)
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        sys.exit(1)