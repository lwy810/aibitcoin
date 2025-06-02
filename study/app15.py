//@version=4
// Squeeze Momentum Indicator [LazyBear]
// Converted to version 4 by TradingView
// This script is an adaptation of John Carter's "TTM Squeeze" volatility indicator,
// as discussed in his book "Mastering the Trade" (chapter 11).

study("Squeeze Momentum Indicator [LazyBear]", shorttitle="SQZMOM_LB")

length = input(20, title="BB Length")
mult = input(2.0, title="BB MultFactor")
lengthKC = input(20, title="KC Length")
multKC = input(1.5, title="KC MultFactor")

useTrueRange = input(true, title="Use TrueRange (KC)")

// Calculate BB
source = close
basis = sma(source, length)
dev = mult * stdev(source, length)
upperBB = basis + dev
lowerBB = basis - dev

// Calculate KC
ma = sma(source, lengthKC)
range = useTrueRange ? tr : (high - low)
rangema = sma(range, lengthKC)
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

sqzOn = (lowerBB > lowerKC) and (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
noSqz = (sqzOn == false) and (sqzOff == false)

val = linreg(source - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close,lengthKC)), lengthKC,0)

bcolor = iff( val > 0,
            iff( val > nz(val[1]), color.lime, color.green),
            iff( val < nz(val[1]), color.red, color.maroon))
scolor = noSqz ? color.blue : sqzOn ? color.black : color.gray
plot(val, color=bcolor, style=plot.style_histogram, linewidth=4)
plot(0, color=scolor, style=plot.style_cross, linewidth=2)