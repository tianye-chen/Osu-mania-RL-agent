// Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
// See the LICENCE file in the repository root for full licence text.

using System;
using System.Net.Sockets;
using System.Collections.Generic;
using System.Linq;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Judgements;
using osu.Game.Rulesets.Mania.Objects;
using osu.Game.Rulesets.Objects;
using osu.Game.Rulesets.Scoring;
using osu.Game.Screens.Play;
using osu.Framework.Logging;

namespace osu.Game.Rulesets.Mania.Scoring
{
    public partial class ManiaScoreProcessor : ScoreProcessor
    {
        private const double combo_base = 4;
        private TcpClient? client;
        private NetworkStream? stream;


        public ManiaScoreProcessor()
            : base(new ManiaRuleset())
        {
            if (client == null)
            {
                initSocketConnection();
            }
        }

        private void initSocketConnection()
        {
            try
            {
                client = new TcpClient("127.0.0.1", 5555);
                stream = client.GetStream();
            }
            catch (Exception e)
            {
                Logger.Log($"Error connecting to socket: {e.Message}", LoggingTarget.Runtime, LogLevel.Error);
            }
        }

        protected void SendHitOverSocket(int hitResult)
        {
            try
            {
                byte[] data = BitConverter.GetBytes(hitResult);
                stream?.Write(data, 0, data.Length);
            }
            catch
            {
                return;
            }
        }

        protected override void Dispose(bool isDisposing)
        {
            base.Dispose(isDisposing);

            stream?.Close();
            client?.Close();
        }

        protected void SendHitResult(HitResult result)
        {
            if ((Clock as IGameplayClock)?.IsRunning != true)
            {
                return;
            }

            switch (result)
            {
                case HitResult.Miss:
                    SendHitOverSocket(0);
                    break;
                case HitResult.Meh:
                    SendHitOverSocket(1);
                    break;
                case HitResult.Ok:
                    SendHitOverSocket(2);
                    break;
                case HitResult.Good:
                    SendHitOverSocket(3);
                    break;
                case HitResult.Great:
                    SendHitOverSocket(4);
                    break;
                case HitResult.Perfect:
                    SendHitOverSocket(5);
                    break;
                default:
                    break;
            }

            Logger.Log($"Hit result sent: {result}");
        }

        ~ManiaScoreProcessor()
        {
            Dispose(false);
        }

        protected override IEnumerable<HitObject> EnumerateHitObjects(IBeatmap beatmap)
            => base.EnumerateHitObjects(beatmap).Order(JudgementOrderComparer.DEFAULT);

        protected override double ComputeTotalScore(double comboProgress, double accuracyProgress, double bonusPortion)
        {
            return 150000 * comboProgress
                   + 850000 * Math.Pow(Accuracy.Value, 2 + 2 * Accuracy.Value) * accuracyProgress
                   + bonusPortion;
        }

        protected override double GetComboScoreChange(JudgementResult result)
        {
            return getBaseComboScoreForResult(result.Type) * Math.Min(Math.Max(0.5, Math.Log(result.ComboAfterJudgement, combo_base)), Math.Log(400, combo_base));
        }

        public override int GetBaseScoreForResult(HitResult result)
        {
            switch (result)
            {
                case HitResult.Perfect:
                    return 305;
            }

            return base.GetBaseScoreForResult(result);
        }

        private int getBaseComboScoreForResult(HitResult result)
        {
            SendHitResult(result);
            switch (result)
            {
                case HitResult.Perfect:
                    return 300;
            }

            return GetBaseScoreForResult(result);
        }

        private class JudgementOrderComparer : IComparer<HitObject>
        {
            public static readonly JudgementOrderComparer DEFAULT = new JudgementOrderComparer();

            public int Compare(HitObject? x, HitObject? y)
            {
                if (ReferenceEquals(x, y)) return 0;
                if (ReferenceEquals(x, null)) return -1;
                if (ReferenceEquals(y, null)) return 1;

                int result = x.GetEndTime().CompareTo(y.GetEndTime());
                if (result != 0)
                    return result;

                // due to the way input is handled in mania, notes take precedence over ticks in judging order.
                if (x is Note && y is not Note) return -1;
                if (x is not Note && y is Note) return 1;

                return x is ManiaHitObject maniaX && y is ManiaHitObject maniaY
                    ? maniaX.Column.CompareTo(maniaY.Column)
                    : 0;
            }
        }
    }
}
