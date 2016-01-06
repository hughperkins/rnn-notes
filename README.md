# RNN Notes

Notes on how [rnn](https://github.com/element-research/rnn) works.  Mostly derivatives from the rnn documentation, but supplemented a bit by reading the source code.  Note that Colah's blog post on LSTM is really good http://colah.github.io/posts/2015-08-Understanding-LSTMs/ .

## Training process flow

### In absence of rnn

- we have a network, in this case an nn.Sequential
  - contains an nn.Linear and an nn.LogSoftMax
- we train on each pair by doing:
  - net:forward(input)
  - (get gradOutput from criterion; criterion works same for backwardsonline or non-rnn usage)
  - net:backward(input, gradOutput)
  - net:updateParameters(learningRate)
- net:forward will do:
```
   Sequential[Module]:forward  (notation: lowest-level class [class containing method]:method name)
   => Sequential[Sequential]:updateOutput
     => Linear[Linear]:updateOutput (calcs output)
     => LogSoftMax[LogSoftMax]:update Output (calcs output)
```
- net:backward will do:
```
  Sequential[Module]:backward
  => Sequential[Sequential]:updateGradInput
    => LogSoftMax[LogSoftMax]:updateGradInput (calc gradInput)
    => Linear[Linear]:updateGradInput (calc gradInput)
  => Sequential[Sequential]:accGradParameters
    => Linear[Linear]:accGradPararameters (calc gradWeight, gradBias)
```
- netUpdateParameters(learningRate) will do:
```
  Sequential[Container]:updateParameters
    => Linear[Module]:updateParameters  (updates weights, bias)
```

### Using backwardsonline

- training looks like:
```
net:train()
net:forget()
net:backwardsonline()
for s=1,seqLength do
  outputs[s] = net:forward(inputs[s])
end
for s=seqLength,1,-1 do
  -- (get gradOutput from criteria, based on inputs[s] and targets[s]
  -- invariant with non-rnn case, so we ignore it here)
  net:backward(inputs[s], gradOutputs[s])
end
net:updateParameters(learningRate)
```

## Class hierarchies

### nn
```
Sequential => Container  => Module
Linear                   => Module
LogSoftMax               => Module

Sequential:
  updateOutput            modules:updateOutput
  updateGradInput         modules:updateGradInput
  accUpdateGradParameters modules:accUpdateGradParameters
  accGradparameters       modules:accGradParameters
  backward                modules:backward

Container:
  zeroGradParameters      modules:zeroGradParameters
  updateParameters        modules:updateParameters
  parameters              concatenate modules:parameters
  training                modules:training
  evaluate                modules:evaluate
  applyToModules

Module:
  updateOutput            return self.output
  updateGradInput         return self.gradInput
  accGradParameters       nothing
  accUpdateGradParameters self:accGradParameters
  forward                 self:updateOutput
  backward                self:updateGradInput, self:accGradParameters
  backwardUpdate          self:updateGradInput, self:accUpdateGradParameters
  zeroGradParameters      zeros parameters()
  updateParameters        adds learningrate * self.parameters()[2] to parameters()[1]
  training                self.train = true
  evaluate                self.train = false
  clone                   clone, via serialize to memory file
  flatten                 flattens all parameters from self and children into single storage
  getParameters

Linear:
  updateOutput            calc output
  updateGradInput         calc gradInput
  addGradParameters       calc gradWeight, gradBias

LogSoftMax:
  updateOutput
  updateGradInput 
```

### rnn

Adds some methods to existing `nn` classes:
```
Module:
  forget                  modules:forget
  remember                modules:remember
  backwardThroughTime     modules:backwardThroughTime
  backwardOnline          modules:backwardOnline
  maxBPTTstep             modules:maxBPTTstep
  stepClone               return self:sharedClone()
```

New classes:
```
Sequencer => AbstractSequencer => Container => Module
Recursor  => AbstractRecurrent => Container => Module
LSTM      => AbstractRecurrent => Container => Module
```

```
Recursor:
  __init(module)                      self.recurrentModule = module; self.modules = {module}
  updateOutput                        getStepModule (clones self.recurrentModule),
                                        stores input, stores output, increments step
  backwardThroughTime
  updateGradInputThroughTime
  accUpdateGradParametersThroughTime
  sharedClone                         return self
  backwardOnline                      AbstractRecurrent.backwardOnline
  forget

AbstractRecurrent:
  getStepModule(step)                 calls self.recurrentModule:stepClone(), stores in self.sharedClones
                                      - stepClone is in `Module`
                                      - for LSTM, is a nop, returns self
  maskZero
  updateGradInput                     self:updateGradInputThroughTime(self.updateGradInputStep, 1)
                                        decrement self.updateGradInputStep
  accGradParameters
  backwardThroughTime                 nop
  updateGradInputThroughTime          nop
  accGradParametersThroughTime        nop
  accUpdateGradParametersThroughTime  nop
  backwardUpdateThroughTime(lr)
  updateParameters(lr)
  recycle(offset)
  forget(offset)
  includingSharedClones(f)
  type(type)
  training
  evaluate
  backwardOnline

LSTM:
  __init                              self.recurrentModule = self:buildModel()
  buildModel
  updateOutput                        self.outputs[step], self.cells[step] = self.recurrentModel:updateOutput(
                                        input, self.outputs[step-1], selfcells[step-1])
  backwardThroughTime
  updateGradInputThroughTime(step)    self.gradInputs[maxSteps-step], self.gradPrevOutput, self.gradCells[step-2] =
                                        self:getStepModule(step):updateGradInput(
                                          {self.inputs[step-1], self.outputs[step-2], self.cells[step-2]},
                                          {self.gradOutputs[step-1], self.gradCells[step-1]})
  accGradParametersThroughTime
  accUpdateGradParameters
```

