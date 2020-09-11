object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'Form1'
  ClientHeight = 561
  ClientWidth = 669
  Color = clBtnFace
  DoubleBuffered = True
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Segoe UI'
  Font.Style = []
  OldCreateOrder = False
  OnCreate = FormCreate
  OnDestroy = FormDestroy
  OnShow = FormShow
  PixelsPerInch = 96
  TextHeight = 15
  object Image1: TImage
    Left = 8
    Top = 41
    Width = 512
    Height = 512
    Proportional = True
    Stretch = True
  end
  object Label1: TLabel
    Left = 526
    Top = 41
    Width = 34
    Height = 15
    Caption = 'Label1'
  end
  object BBOneEpoch: TBitBtn
    Left = 8
    Top = 8
    Width = 120
    Height = 25
    Caption = 'Train 1 Epoch'
    TabOrder = 0
    OnClick = BBOneEpochClick
  end
  object BBOneHundredEpochs: TBitBtn
    Left = 134
    Top = 8
    Width = 120
    Height = 25
    Caption = 'Train 100 Epochs'
    TabOrder = 1
    OnClick = BBOneHundredEpochsClick
  end
  object BBReset: TBitBtn
    Left = 400
    Top = 8
    Width = 120
    Height = 25
    Caption = 'Reset'
    TabOrder = 2
    OnClick = BBResetClick
  end
end
