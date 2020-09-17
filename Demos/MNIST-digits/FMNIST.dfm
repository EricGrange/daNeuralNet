object FormMNIST: TFormMNIST
  Left = 0
  Top = 0
  Caption = 'MNIST digits'
  ClientHeight = 462
  ClientWidth = 1008
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Segoe UI'
  Font.Style = []
  OldCreateOrder = False
  OnCreate = FormCreate
  DesignSize = (
    1008
    462)
  PixelsPerInch = 96
  TextHeight = 15
  object Image: TImage
    Left = 248
    Top = 112
    Width = 105
    Height = 105
    Proportional = True
    Stretch = True
  end
  object Splitter1: TSplitter
    Left = 0
    Top = 0
    Height = 462
    ExplicitLeft = 696
    ExplicitTop = 88
    ExplicitHeight = 100
  end
  object Label1: TLabel
    Left = 17
    Top = 56
    Width = 9
    Height = 15
    Caption = '...'
  end
  object BUOneEpoch: TButton
    Left = 17
    Top = 17
    Width = 121
    Height = 25
    Caption = 'Run 1 Epoch'
    TabOrder = 0
    OnClick = BUOneEpochClick
  end
  object ListBox: TListBox
    Left = 17
    Top = 112
    Width = 225
    Height = 342
    Anchors = [akLeft, akTop, akBottom]
    ItemHeight = 15
    TabOrder = 1
    OnClick = ListBoxClick
    ExplicitHeight = 310
  end
  object Chart1: TChart
    Left = 376
    Top = 8
    Width = 617
    Height = 445
    Legend.Visible = False
    MarginBottom = 0
    MarginLeft = 0
    MarginRight = 0
    MarginTop = 0
    Title.Text.Strings = (
      'TChart')
    Title.Visible = False
    View3D = False
    BevelOuter = bvNone
    TabOrder = 2
    Anchors = [akLeft, akTop, akRight, akBottom]
    ExplicitHeight = 413
    DefaultCanvas = 'TTeeCanvas3D'
    ColorPaletteIndex = 13
    object SETestScore: TAreaSeries
      Gradient.EndColor = 16768220
      Gradient.Visible = True
      SeriesColor = clAqua
      AreaChartBrush.Color = clGray
      AreaChartBrush.BackColor = clDefault
      AreaChartBrush.Gradient.EndColor = 16768220
      AreaChartBrush.Gradient.Visible = True
      AreaLinesPen.Visible = False
      DrawArea = True
      Pointer.InflateMargins = True
      Pointer.Style = psRectangle
      Pointer.Visible = False
      XValues.Name = 'X'
      XValues.Order = loAscending
      YValues.Name = 'Y'
      YValues.Order = loNone
    end
    object SETrainScore: TLineSeries
      Brush.BackColor = clDefault
      LinePen.Width = 2
      Pointer.InflateMargins = True
      Pointer.Style = psRectangle
      XValues.Name = 'X'
      XValues.Order = loAscending
      YValues.Name = 'Y'
      YValues.Order = loNone
    end
  end
  object BUReset: TButton
    Left = 278
    Top = 17
    Width = 75
    Height = 25
    Caption = 'Reset'
    TabOrder = 3
    OnClick = BUResetClick
  end
  object TeeGDIPlus1: TTeeGDIPlus
    AntiAliasText = gpfBest
    TeePanel = Chart1
    Left = 304
    Top = 240
  end
end
